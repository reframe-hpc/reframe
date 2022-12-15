# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import math
import sys
import time

import reframe.core.runtime as rt
import reframe.utility as util
from reframe.core.exceptions import (FailureLimitError,
                                     SkipTestError,
                                     TaskDependencyError,
                                     TaskExit)
from reframe.core.logging import getlogger, level_from_str
from reframe.core.pipeline import (CompileOnlyRegressionTest,
                                   RunOnlyRegressionTest)
from reframe.frontend.executors import (ExecutionPolicy, RegressionTask,
                                        TaskEventListener, ABORT_REASONS)


def _get_partition_name(task, phase='run'):
    if (task.check.local or
        (phase == 'build' and task.check.build_locally)):
        return '_rfm_local'
    else:
        return task.check.current_partition.fullname


def _cleanup_all(tasks, *args, **kwargs):
    for task in tasks:
        if task.ref_count == 0:
            with contextlib.suppress(TaskExit):
                task.cleanup(*args, **kwargs)

    # Remove cleaned up tests
    tasks[:] = [t for t in tasks if t.ref_count]


def _print_perf(task):
    '''Get performance info of the current task.'''

    perfvars = task.testcase.check.perfvalues
    level = level_from_str(
        rt.runtime().get_option('general/0/perf_info_level')
    )
    for key, info in perfvars.items():
        name = key.split(':')[-1]
        getlogger().log(level,
                        f'P: {name}: {info[0]} {info[4]} '
                        f'(r:{info[1]}, l:{info[2]}, u:{info[3]})')


class _PollController:
    SLEEP_MIN = 0.1
    SLEEP_MAX = 10
    SLEEP_INC_RATE = 1.1

    def __init__(self):
        self._num_polls = 0
        self._sleep_duration = None
        self._t_init = None

    def reset_snooze_time(self):
        self._sleep_duration = self.SLEEP_MIN

    def snooze(self):
        if self._num_polls == 0:
            self._t_init = time.time()

        t_elapsed = time.time() - self._t_init
        self._num_polls += 1
        poll_rate = self._num_polls / t_elapsed if t_elapsed else math.inf
        getlogger().debug2(
            f'Poll rate control: sleeping for {self._sleep_duration}s '
            f'(current poll rate: {poll_rate} polls/s)'
        )
        time.sleep(self._sleep_duration)
        self._sleep_duration = min(
            self._sleep_duration*self.SLEEP_INC_RATE, self.SLEEP_MAX
        )


class SerialExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):
        super().__init__()

        self._pollctl = _PollController()

        # Index tasks by test cases
        self._task_index = {}

        # Tasks that have finished, but have not performed their cleanup phase
        self._retired_tasks = []
        self.task_listeners.append(self)

    def runcase(self, case):
        super().runcase(case)
        check, partition, environ = case
        task = RegressionTask(case, self.task_listeners)
        self.printer.status('RUN', task.info())
        self._task_index[case] = task
        self.stats.add_task(task)
        try:
            # Do not run test if any of its dependencies has failed
            # NOTE: Restored dependencies are not in the task_index
            if any(self._task_index[c].failed
                   for c in case.deps if c in self._task_index):
                raise TaskDependencyError('dependencies failed')

            if any(self._task_index[c].skipped
                   for c in case.deps if c in self._task_index):

                # We raise the SkipTestError here and catch it immediately in
                # order for `skip()` to get the correct exception context.
                try:
                    raise SkipTestError('skipped due to skipped dependencies')
                except SkipTestError as e:
                    task.skip()
                    raise TaskExit from e

            task.setup(task.testcase.partition,
                       task.testcase.environ,
                       sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                       sched_options=self.sched_options)

            task.compile()
            task.compile_wait()
            task.run()

            # Pick the right scheduler
            if task.check.local:
                sched = self.local_scheduler
            else:
                sched = partition.scheduler

            self._pollctl.reset_snooze_time()
            while True:
                sched.poll(task.check.job)
                if task.run_complete():
                    break

                self._pollctl.snooze()

            task.run_wait()
            if not self.skip_sanity_check:
                task.sanity()

            if not self.skip_performance_check:
                task.performance()

            self._retired_tasks.append(task)
            task.finalize()
        except TaskExit:
            return
        except ABORT_REASONS as e:
            task.abort(e)
            raise
        except BaseException:
            task.fail(sys.exc_info())

    def on_task_setup(self, task):
        pass

    def on_task_run(self, task):
        pass

    def on_task_compile(self, task):
        pass

    def on_task_exit(self, task):
        pass

    def on_task_compile_exit(self, task):
        pass

    def on_task_skip(self, task):
        msg = str(task.exc_info[1])
        self.printer.status('SKIP', msg, just='right')

    def on_task_failure(self, task):
        self._num_failed_tasks += 1
        msg = f'{task.info()}'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self.printer.status('FAIL', msg, just='right')

        _print_perf(task)
        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                         'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().info(f'==> test failed during {task.failed_stage!r}: '
                         f'test staged in {task.check.stagedir!r}')
        getlogger().verbose(f'==> {timings}')
        if self._num_failed_tasks >= self.max_failures:
            raise FailureLimitError(
                f'maximum number of failures ({self.max_failures}) reached'
            )

    def on_task_success(self, task):
        msg = f'{task.info()}'
        self.printer.status('OK', msg, just='right')
        _print_perf(task)
        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                         'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().verbose(f'==> {timings}')

        # Update reference count of dependencies
        for c in task.testcase.deps:
            # NOTE: Restored dependencies are not in the task_index
            if c in self._task_index:
                self._task_index[c].ref_count -= 1

        _cleanup_all(self._retired_tasks, not self.keep_stage_files)

    def exit(self):
        # Clean up all remaining tasks
        _cleanup_all(self._retired_tasks, not self.keep_stage_files)


class AsynchronousExecutionPolicy(ExecutionPolicy, TaskEventListener):
    '''The asynchronous execution policy.'''

    def __init__(self):
        super().__init__()

        self._pollctl = _PollController()

        # Index tasks by test cases
        self._task_index = {}

        # A set of all the current tasks. We use an ordered set here, because
        # we want to preserve the order of the tasks.
        self._current_tasks = util.OrderedSet()

        # Quick look up for the partition schedulers including the
        # `_rfm_local` pseudo-partition
        self._schedulers = {
            '_rfm_local': self.local_scheduler
        }

        # Tasks per partition
        self._partition_tasks = {
            '_rfm_local': util.OrderedSet()
        }

        # Retired tasks that need to be cleaned up
        self._retired_tasks = []

        # Job limit per partition
        self._max_jobs = {
            '_rfm_local': rt.runtime().get_option('systems/0/max_local_jobs')
        }
        self._pipeline_statistics = rt.runtime().get_option(
            'systems/0/dump_pipeline_progress'
        )
        self.task_listeners.append(self)

    def _init_pipeline_progress(self, num_tasks):
        self._pipeline_progress = {
            'startup': [(num_tasks, 0)],
            'ready_compile': [(0, 0)],
            'compiling': [(0, 0)],
            'ready_run': [(0, 0)],
            'running': [(0, 0)],
            'completing': [(0, 0)],
            'retired': [(0, 0)],
            'completed': [(0, 0)],
            'fail': [(0, 0)],
            'skip': [(0, 0)]
        }
        self._pipeline_step = 0
        self._t_pipeline_start = time.time()

    def _update_pipeline_progress(self, old_state, new_state, num_tasks=1):
        timestamp = time.time() - self._t_pipeline_start
        for state in self._pipeline_progress:
            count = self._pipeline_progress[state][self._pipeline_step][0]
            if old_state != new_state:
                if state == old_state:
                    count -= num_tasks
                elif state == new_state:
                    count += num_tasks

            self._pipeline_progress[state].append((count, timestamp))

        self._pipeline_step += 1

    def _dump_pipeline_progress(self, filename):
        import reframe.utility.jsonext as jsonext

        with open(filename, 'w') as fp:
            jsonext.dump(self._pipeline_progress, fp, indent=2)

    def runcase(self, case):
        super().runcase(case)
        check, partition, environ = case
        self._schedulers[partition.fullname] = partition.scheduler

        # Set partition-based counters, if not set already
        self._partition_tasks.setdefault(partition.fullname, util.OrderedSet())
        self._max_jobs.setdefault(partition.fullname, partition.max_jobs)

        task = RegressionTask(case, self.task_listeners)
        self._task_index[case] = task
        self.stats.add_task(task)
        getlogger().debug2(
            f'Added {check.name} on {partition.fullname} '
            f'using {environ.name}'
        )
        self._current_tasks.add(task)

    def exit(self):
        if self._pipeline_statistics:
            self._init_pipeline_progress(len(self._current_tasks))

        self._pollctl.reset_snooze_time()
        while self._current_tasks:
            try:
                self._poll_tasks()
                num_running = sum(
                    1 if t.state in ('running', 'compiling') else 0
                    for t in self._current_tasks
                )

                timeout = rt.runtime().get_option(
                    'general/0/pipeline_timeout'
                )

                self._advance_all(self._current_tasks, timeout)
                if self._pipeline_statistics:
                    num_retired = len(self._retired_tasks)

                _cleanup_all(self._retired_tasks, not self.keep_stage_files)
                if self._pipeline_statistics:
                    num_retired_actual = num_retired - len(self._retired_tasks)

                    # Some tests might not be cleaned up because they are
                    # waiting for dependencies or because their dependencies
                    # have failed.
                    self._update_pipeline_progress(
                        'retired', 'completed', num_retired_actual
                    )

                if num_running:
                    self._pollctl.snooze()
            except ABORT_REASONS as e:
                self._failall(e)
                raise

        if self._pipeline_statistics:
            self._dump_pipeline_progress('pipeline-progress.json')

    def _poll_tasks(self):
        for partname, sched in self._schedulers.items():
            jobs = []
            for t in self._partition_tasks[partname]:
                if t.state == 'compiling':
                    jobs.append(t.check.build_job)
                elif t.state == 'running':
                    jobs.append(t.check.job)

            sched.poll(*jobs)

    def _exec_stage(self, task, stage_methods):
        '''Execute a series of pipeline stages.

        Return True on success, False otherwise.
        '''

        try:
            for stage in stage_methods:
                stage()
        except TaskExit:
            self._current_tasks.remove(task)
            if task.check.current_partition:
                partname = task.check.current_partition.fullname
            else:
                partname = None

            # Remove tasks from the partition tasks if there
            with contextlib.suppress(KeyError):
                self._partition_tasks['_rfm_local'].remove(task)
                if partname:
                    self._partition_tasks[partname].remove(task)

            return False
        else:
            return True

    def _advance_all(self, tasks, timeout=None):
        t_init = time.time()
        num_progressed = 0

        getlogger().debug2(f'Current tests: {len(tasks)}')

        # We take a snapshot of the tasks to advance by doing a shallow copy,
        # since the tasks may removed by the individual advance functions.
        for t in list(tasks):
            old_state = t.state
            bump_state = getattr(self, f'_advance_{t.state}')
            num_progressed += bump_state(t)
            new_state = t.state

            t_elapsed = time.time() - t_init
            if timeout and t_elapsed > timeout and num_progressed:
                break

            if self._pipeline_statistics:
                self._update_pipeline_progress(old_state, new_state, 1)

        getlogger().debug2(f'Bumped {num_progressed} test(s)')

    def _advance_startup(self, task):
        if self.deps_skipped(task):
            try:
                raise SkipTestError('skipped due to skipped dependencies')
            except SkipTestError as e:
                task.skip()
                self._current_tasks.remove(task)
                return 1
        elif self.deps_succeeded(task):
            try:
                self.printer.status('RUN', task.info())
                task.setup(task.testcase.partition,
                           task.testcase.environ,
                           sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                           sched_options=self.sched_options)
            except TaskExit:
                self._current_tasks.remove(task)
                return 1

            if isinstance(task.check, RunOnlyRegressionTest):
                # All tests should execute all the pipeline stages, even if
                # they are no-ops
                self._exec_stage(task, [task.compile,
                                        task.compile_complete,
                                        task.compile_wait])

            return 1
        elif self.deps_failed(task):
            exc = TaskDependencyError('dependencies failed')
            task.fail((type(exc), exc, None))
            self._current_tasks.remove(task)
            return 1
        else:
            # Not all dependencies have finished yet
            getlogger().debug2(f'{task.info()} waiting for dependencies')
            return 0

    def _advance_ready_compile(self, task):
        partname = _get_partition_name(task, phase='build')
        max_jobs = self._max_jobs[partname]
        if len(self._partition_tasks[partname]) < max_jobs:
            if self._exec_stage(task, [task.compile]):
                self._partition_tasks[partname].add(task)

            return 1

        getlogger().debug2(f'Hit the max job limit of {partname}: {max_jobs}')
        return 0

    def _advance_compiling(self, task):
        partname = _get_partition_name(task, phase='build')
        try:
            if task.compile_complete():
                task.compile_wait()
                self._partition_tasks[partname].remove(task)
                if isinstance(task.check, CompileOnlyRegressionTest):
                    # All tests should pass from all the pipeline stages,
                    # even if they are no-ops
                    self._exec_stage(task, [task.run,
                                            task.run_complete,
                                            task.run_wait])

                return 1
            else:
                return 0
        except TaskExit:
            self._partition_tasks[partname].remove(task)
            self._current_tasks.remove(task)
            return 1

    def _advance_ready_run(self, task):
        partname = _get_partition_name(task, phase='run')
        max_jobs = self._max_jobs[partname]
        if len(self._partition_tasks[partname]) < max_jobs:
            if self._exec_stage(task, [task.run]):
                self._partition_tasks[partname].add(task)

            return 1

        getlogger().debug2(f'Hit the max job limit of {partname}: {max_jobs}')
        return 0

    def _advance_running(self, task):
        partname = _get_partition_name(task, phase='run')
        try:
            if task.run_complete():
                if self._exec_stage(task, [task.run_wait]):
                    self._partition_tasks[partname].remove(task)

                return 1
            else:
                return 0

        except TaskExit:
            self._partition_tasks[partname].remove(task)
            self._current_tasks.remove(task)
            return 1

    def _advance_completing(self, task):
        try:
            if not self.skip_sanity_check:
                task.sanity()

            if not self.skip_performance_check:
                task.performance()

            task.finalize()
            self._retired_tasks.append(task)
            self._current_tasks.remove(task)
        except TaskExit:
            self._current_tasks.remove(task)
        finally:
            return 1

    def deps_failed(self, task):
        # NOTE: Restored dependencies are not in the task_index
        return any(self._task_index[c].failed
                   for c in task.testcase.deps if c in self._task_index)

    def deps_succeeded(self, task):
        # NOTE: Restored dependencies are not in the task_index
        return all(self._task_index[c].succeeded
                   for c in task.testcase.deps if c in self._task_index)

    def deps_skipped(self, task):
        # NOTE: Restored dependencies are not in the task_index
        return any(self._task_index[c].skipped
                   for c in task.testcase.deps if c in self._task_index)

    def _failall(self, cause):
        '''Mark all tests as failures'''
        getlogger().debug2(f'Aborting all tasks due to {type(cause).__name__}')
        for task in self._current_tasks:
            with contextlib.suppress(FailureLimitError):
                task.abort(cause)

    # These function can be useful for tracking statistics of the framework,
    # such as number of tests that have finished setup etc.
    def on_task_setup(self, task):
        pass

    def on_task_run(self, task):
        pass

    def on_task_compile(self, task):
        pass

    def on_task_exit(self, task):
        self._pollctl.reset_snooze_time()

    def on_task_compile_exit(self, task):
        self._pollctl.reset_snooze_time()

    def on_task_skip(self, task):
        msg = str(task.exc_info[1])
        self.printer.status('SKIP', msg, just='right')

    def on_task_failure(self, task):
        self._num_failed_tasks += 1
        msg = f'{task.info()}'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self.printer.status('FAIL', msg, just='right')

        _print_perf(task)
        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                         'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().info(f'==> test failed during {task.failed_stage!r}: '
                         f'test staged in {task.check.stagedir!r}')
        getlogger().verbose(f'==> {timings}')
        if self._num_failed_tasks >= self.max_failures:
            raise FailureLimitError(
                f'maximum number of failures ({self.max_failures}) reached'
            )

    def on_task_success(self, task):
        msg = f'{task.info()}'
        self.printer.status('OK', msg, just='right')
        _print_perf(task)
        timings = task.pipeline_timings(['setup',
                                         'compile_complete',
                                         'run_complete',
                                         'sanity',
                                         'performance',
                                         'total'])
        getlogger().verbose(f'==> {timings}')
        for c in task.testcase.deps:
            # NOTE: Restored dependencies are not in the task_index
            if c in self._task_index:
                self._task_index[c].ref_count -= 1
