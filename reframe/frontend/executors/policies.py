# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import random
import sys
import time

import reframe.core.runtime as rt
import reframe.utility as util
import reframe.utility.color as color
from reframe.core.exceptions import (ConfigError,
                                     FailureLimitError,
                                     RunSessionTimeout,
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
        val, ref, lower, upper, unit, result = info
        name = key.split(':')[-1]
        
        # Build reference info string only if all three reference value components are defined
        if ref is not None and lower is not None and upper is not None:
            msg = f'P: {name}: {val} {unit} (r:{ref}, l:{lower}, u:{upper})'
        else:
            msg = f'P: {name}: {val} {unit}'
            
        if result == 'xfail':
            msg = color.colorize(msg, color.MAGENTA)
        elif result == 'fail' or result == 'xpass':
            msg = color.colorize(msg, color.RED)

        getlogger().log(level, msg)


def _print_pipeline_timings(task):
    timings = task.pipeline_timings(['setup',
                                     'compile_complete',
                                     'run_complete',
                                     'sanity',
                                     'performance',
                                     'total'])
    getlogger().verbose(f'==> {timings}')


class _PollController:
    def _validate_poll_params(self):
        def _check_positive(x, name):
            if x <= 0:
                raise ConfigError(f'{name} must be a positive number')

        _check_positive(self._poll_rate_min, 'minimum poll rate')
        _check_positive(self._poll_rate_max, 'maximum poll rate')
        if self._poll_rate_max < self._poll_rate_min:
            raise ConfigError('maximum poll rate must be greater or equal to '
                              'minimum poll rate')

        if self._poll_rate_decay < 0 or self._poll_rate_decay > 1:
            raise ConfigError('poll rate decay must be in range [0,1]')

        if self._poll_randomize_range_ms is not None:
            left, right = self._poll_randomize_range_ms
            if left > 0:
                raise ConfigError('left boundary of poll randomization range '
                                  'must be a negative integer or zero')

            if right < 0:
                raise ConfigError('right boundary of poll randomization range '
                                  'must be a positive integer or zero')

    def __init__(self):
        get_option = rt.runtime().get_option
        self._poll_rate_max = get_option('general/0/poll_rate_max')
        self._poll_rate_min = get_option('general/0/poll_rate_min')
        self._poll_rate_decay = get_option('general/0/poll_rate_decay')
        self._poll_randomize_range_ms = get_option(
            'general/0/poll_randomize_ms'
        )
        self._poll_count_total = 0
        self._poll_count_interval = 0
        self._validate_poll_params()
        self._t_start = None
        self._t_last_reset = None
        self._desired_poll_rate = self._poll_rate_max

    def reset_poll_rate(self):
        getlogger().debug2('[P] reset poll rate')
        self._poll_count_interval = 0
        self._desired_poll_rate = self._poll_rate_max
        self._t_last_reset = time.time()

    def _poll_rate(self):
        now = time.time()
        return (self._poll_count_total / (now - self._t_start),
                self._poll_count_interval / (now - self._t_last_reset))

    def snooze(self):
        if self._poll_count_total == 0:
            self._t_start = time.time()
            self._t_last_reset = self._t_start
            dt_sleep = 1. / self._desired_poll_rate
        else:
            dt_next_interval = time.time() - self._t_last_reset
            dt_sleep = (self._poll_count_interval + 1) / self._desired_poll_rate - dt_next_interval  # noqa: E501

        if self._poll_randomize_range_ms:
            sleep_eps = random.uniform(*self._poll_randomize_range_ms)
            dt_sleep += sleep_eps / 1000

        # Make sure sleep time positive
        dt_sleep = max(0, dt_sleep)
        time.sleep(dt_sleep)

        self._poll_count_total += 1
        self._poll_count_interval += 1
        poll_rate_global, poll_rate_curr = self._poll_rate()
        getlogger().debug2(f'[P] sleep_time={dt_sleep:.6f}, '
                           f'pr_desired={self._desired_poll_rate:.6f}, '
                           f'pr_current={poll_rate_curr:.6f}, '
                           f'pr_global={poll_rate_global:.6f}')
        self._desired_poll_rate = max(
            self._desired_poll_rate * (1 - self._poll_rate_decay),
            self._poll_rate_min
        )


class _PolicyEventListener(TaskEventListener):
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
        msg = f'{task.info()} [{task.exc_info[1]}]'
        self.printer.status('SKIP', msg, just='right')

    def on_task_abort(self, task):
        msg = f'{task.info()}'
        self.printer.status('ABORT', msg, just='right')

    def on_task_xfailure(self, task):
        msg = f'{task.info()} [{task.exc_info[1]}]'
        self.printer.status('XFAIL', msg, just='right')
        _print_perf(task)
        if task.failed_stage == 'sanity':
            # Dry-run the performance stage to trigger performance logging
            task.performance(dry_run=True)

        _print_pipeline_timings(task)

    def on_task_failure(self, task):
        self._num_failed_tasks += 1
        msg = f'{task.info()}'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self.printer.status('FAIL', msg, just='right')

        _print_perf(task)
        if task.failed_stage == 'sanity':
            # Dry-run the performance stage to trigger performance logging
            task.performance(dry_run=True)

        getlogger().info(f'==> test failed during {task.failed_stage!r}: '
                         f'test staged in {task.check.stagedir!r}')
        _print_pipeline_timings(task)
        if self._num_failed_tasks >= self.max_failures:
            raise FailureLimitError(
                f'maximum number of failures ({self.max_failures}) reached'
            )

    def on_task_xsuccess(self, task):
        msg = f'{task.info()}'
        self.printer.status('XPASS', msg, just='right')
        _print_perf(task)
        if task.failed_stage == 'sanity':
            # Dry-run the performance stage to trigger performance logging
            task.performance(dry_run=True)

        _print_pipeline_timings(task)

    def on_task_success(self, task):
        msg = f'{task.info()}'
        self.printer.status('OK', msg, just='right')
        _print_perf(task)
        _print_pipeline_timings(task)

        # Update reference count of dependencies
        for c in task.testcase.deps:
            # NOTE: Restored dependencies are not in the task_index
            if c in self._task_index:
                self._task_index[c].ref_count -= 1


class SerialExecutionPolicy(ExecutionPolicy, _PolicyEventListener):
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
        check, partition, _ = case
        task = RegressionTask(case, self.task_listeners)
        if check.is_dry_run():
            self.printer.status('DRY', task.info())
        else:
            self.printer.status('RUN', task.info())

        self._task_index[case] = task
        self.stats.add_task(task)
        try:
            # Do not run test if any of its dependencies has failed
            # NOTE: Restored dependencies are not in the task_index
            if any(self._task_index[c].failed
                   for c in case.deps if c in self._task_index):
                task.skip_from_deps()
                raise TaskExit

            if any(self._task_index[c].skipped or self._task_index[c].xfailed
                   for c in case.deps if c in self._task_index):
                task.do_skip('skipped due to skipped dependencies')
                raise TaskExit

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

            self._pollctl.reset_poll_rate()
            while True:
                if not self.dry_run_mode:
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

    def on_task_success(self, task):
        super().on_task_success(task)
        _cleanup_all(self._retired_tasks, not self.keep_stage_files)
        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def on_task_failure(self, task):
        super().on_task_failure(task)
        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def exit(self):
        # Clean up all remaining tasks
        _cleanup_all(self._retired_tasks, not self.keep_stage_files)


class AsynchronousExecutionPolicy(ExecutionPolicy, _PolicyEventListener):
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
            'general/0/dump_pipeline_progress'
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

        self._pollctl.reset_poll_rate()
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

                if self.timeout_expired():
                    raise RunSessionTimeout(
                        'maximum session duration exceeded'
                    )

                if num_running:
                    self._pollctl.snooze()
            except ABORT_REASONS as e:
                self._abortall(e)
                raise

        if self._pipeline_statistics:
            self._dump_pipeline_progress('pipeline-progress.json')

    def _poll_tasks(self):
        if self.dry_run_mode:
            return

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

            if self._pipeline_statistics:
                self._update_pipeline_progress(old_state, new_state, 1)

            t_elapsed = time.time() - t_init
            if timeout and t_elapsed > timeout and num_progressed:
                break

        getlogger().debug2(f'Bumped {num_progressed} test(s)')

    def _advance_startup(self, task):
        if self.deps_skipped(task):
            task.do_skip('skipped due to skipped dependencies')
            self._current_tasks.remove(task)
            return 1
        elif self.deps_succeeded(task):
            try:
                if task.check.is_dry_run():
                    self.printer.status('DRY', task.info())
                else:
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
            task.skip_from_deps()
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
            return 1
        except TaskExit:
            self._current_tasks.remove(task)
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
        return any(self._task_index[c].skipped or self._task_index[c].xfailed
                   for c in task.testcase.deps if c in self._task_index)

    def _abortall(self, cause):
        '''Mark all tests as failures'''

        getlogger().debug2(f'Aborting all tasks due to {type(cause).__name__}')
        for task in self._current_tasks:
            with contextlib.suppress(FailureLimitError):
                task.abort(cause)

    def on_task_exit(self, task):
        self._pollctl.reset_poll_rate()

    def on_task_compile_exit(self, task):
        self._pollctl.reset_poll_rate()
