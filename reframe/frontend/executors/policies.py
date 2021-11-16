# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import functools
import itertools
import math
import sys
import time

import reframe.core.runtime as rt
from reframe.core.exceptions import (FailureLimitError,
                                     SkipTestError,
                                     TaskDependencyError,
                                     TaskExit)
from reframe.core.logging import getlogger
from reframe.core.pipeline import (CompileOnlyRegressionTest,
                                   RunOnlyRegressionTest)
from reframe.frontend.executors import (ExecutionPolicy, RegressionTask,
                                        TaskEventListener, ABORT_REASONS)


def countall(d):
    res = 0
    for (q1, q2) in d.values():
        res += len(q1)
        res += len(q2)

    return res


def _cleanup_all(tasks, *args, **kwargs):
    for task in tasks:
        if task.ref_count == 0:
            with contextlib.suppress(TaskExit):
                task.cleanup(*args, **kwargs)

    # Remove cleaned up tests
    tasks[:] = [t for t in tasks if t.ref_count]


class _PollController:
    SLEEP_MIN = 0.1
    SLEEP_MAX = 10
    SLEEP_INC_RATE = 1.1

    def __init__(self):
        self._num_polls = 0
        self._num_tasks = 0
        self._sleep_duration = None
        self._t_init = None

    def running_tasks(self, num_tasks):
        if self._sleep_duration is None:
            self._sleep_duration = self.SLEEP_MIN

        if self._num_polls == 0:
            self._t_init = time.time()
        else:
            if self._num_tasks != num_tasks:
                self._sleep_duration = self.SLEEP_MIN
            else:
                self._sleep_duration = min(
                    self._sleep_duration*self.SLEEP_INC_RATE, self.SLEEP_MAX
                )

        self._num_tasks = num_tasks
        return self

    def snooze(self):
        t_elapsed = time.time() - self._t_init
        self._num_polls += 1
        poll_rate = self._num_polls / t_elapsed if t_elapsed else math.inf
        getlogger().debug2(
            f'Poll rate control: sleeping for {self._sleep_duration}s '
            f'(current poll rate: {poll_rate} polls/s)'
        )
        time.sleep(self._sleep_duration)


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

        self.printer.status(
            'RUN', '%s on %s using %s' %
            (check.name, partition.fullname, environ.name)
        )
        task = RegressionTask(case, self.task_listeners)
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

            partname = task.testcase.partition.fullname
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

            while True:
                sched.poll(task.check.job)
                if task.run_complete():
                    break

                self._pollctl.running_tasks(1).snooze()

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
        timings = task.pipeline_timings(['compile_complete',
                                         'run_complete',
                                         'total'])
        msg = f'{task.check.info()} [{timings}]'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self.printer.status('FAIL', msg, just='right')

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
        timings = task.pipeline_timings(['compile_complete',
                                         'run_complete',
                                         'total'])
        msg = f'{task.check.info()} [{timings}]'
        self.printer.status('OK', msg, just='right')
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
    def __init__(self):

        super().__init__()

        self._pollctl = _PollController()

        # Index tasks by test cases
        self._task_index = {}

        # Tasks that are waiting for dependencies
        self._waiting_tasks = []

        # Tasks ready to be compiled per partition
        self._ready_to_compile_tasks = {}

        # All tasks currently in their build phase per partition
        self._compiling_tasks = {}

        # Tasks ready to run per partition
        self._ready_to_run_tasks = {}

        # All tasks currently in their run phase per partition
        self._running_tasks = {}

        # Tasks that need to be finalized
        self._completed_tasks = []

        # Retired tasks that need to be cleaned up
        self._retired_tasks = []

        # Job limit per partition
        self._max_jobs = {}

        # Max jobs spawned by the reframe thread
        self._rfm_max_jobs = rt.runtime().get_option(f'systems/0/rfm_max_jobs')

        # Keep a reference to all the partitions
        self._partitions = set()

        self.task_listeners.append(self)

    def _remove_from_running(self, task):
        getlogger().debug2(
            f'Removing task from the running list: {task.testcase}'
        )
        try:
            partname = task.check.current_partition.fullname
            self._running_tasks[partname][self.local_index(task, phase='run')].remove(task)
        except (ValueError, AttributeError, KeyError):
            getlogger().debug2('Task was not running')
            pass

    def _remove_from_building(self, task):
        getlogger().debug2(
            f'Removing task from the building list: {task.testcase}'
        )
        try:
            partname = task.check.current_partition.fullname
            self._compiling_tasks[partname][self.local_index(task, phase='compile')].remove(task)
        except (ValueError, AttributeError, KeyError):
            getlogger().debug2('Task was not building')
            pass

    # FIXME: The following functions are very similar and they are also reused
    # in the serial policy; we should refactor them
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

    def local_index(self, task, phase='run'):
        return (
            task.check.local or
            (phase == 'compile' and task.check.build_locally)
        )

    def on_task_setup(self, task):
        partname = task.check.current_partition.fullname
        if (isinstance(task.check, RunOnlyRegressionTest)):
            self._ready_to_run_tasks[partname][self.local_index(task, phase='run')].append(task)
        else:
            self._ready_to_compile_tasks[partname][self.local_index(task, phase='compile')].append(task)

    def on_task_run(self, task):
        partname = task.check.current_partition.fullname
        self._running_tasks[partname][self.local_index(task, phase='run')].append(task)

    def on_task_compile(self, task):
        partname = task.check.current_partition.fullname
        self._compiling_tasks[partname][self.local_index(task, phase='compile')].append(task)

    def on_task_skip(self, task):
        # Remove the task from the running list if it was skipped after the
        # run phase
        if task.check.current_partition:
            partname = task.check.current_partition.fullname
            if task.failed_stage in ('run_complete', 'run_wait'):
                self._running_tasks[partname][self.local_index(task, phase='run')].remove(task)

            if task.failed_stage in ('compile_complete', 'compile_wait'):
                self._compiling_tasks[partname][self.local_index(task, phase='compile')].remove(task)

        msg = str(task.exc_info[1])
        self.printer.status('SKIP', msg, just='right')

    def on_task_failure(self, task):
        if task.aborted:
            return

        self._num_failed_tasks += 1
        msg = f'{task.check.info()} [{task.pipeline_timings_basic()}]'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self._remove_from_running(task)
            self._remove_from_building(task)
            self.printer.status('FAIL', msg, just='right')

        stagedir = task.check.stagedir
        if not stagedir:
            stagedir = '<not available>'

        getlogger().info(f'==> test failed during {task.failed_stage!r}: '
                         f'test staged in {stagedir!r}')
        getlogger().verbose(f'==> timings: {task.pipeline_timings_all()}')
        if self._num_failed_tasks >= self.max_failures:
            raise FailureLimitError(
                f'maximum number of failures ({self.max_failures}) reached'
            )

    def on_task_success(self, task):
        msg = f'{task.check.info()} [{task.pipeline_timings_basic()}]'
        self.printer.status('OK', msg, just='right')
        getlogger().verbose(f'==> timings: {task.pipeline_timings_all()}')

        # Update reference count of dependencies
        for c in task.testcase.deps:
            # NOTE: Restored dependencies are not in the task_index
            if c in self._task_index:
                self._task_index[c].ref_count -= 1

        self._retired_tasks.append(task)

    def on_task_exit(self, task):
        task.run_wait()
        self._remove_from_running(task)
        self._completed_tasks.append(task)

    def on_task_compile_exit(self, task):
        task.compile_wait()
        self._remove_from_building(task)
        partname = task.check.current_partition.fullname
        if (isinstance(task.check, CompileOnlyRegressionTest)):
            self._completed_tasks.append(task)
        else:
            self._ready_to_run_tasks[partname][self.local_index(task, phase='run')].append(task)

    def _setup_task(self, task):
        if self.deps_skipped(task):
            try:
                raise SkipTestError('skipped due to skipped dependencies')
            except SkipTestError as e:
                task.skip()
                return False
        elif self.deps_succeeded(task):
            try:
                task.setup(task.testcase.partition,
                           task.testcase.environ,
                           sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                           sched_options=self.sched_options)
            except TaskExit:
                return False
            else:
                return True
        elif self.deps_failed(task):
            exc = TaskDependencyError('dependencies failed')
            task.fail((type(exc), exc, None))
            return False
        else:
            # Not all dependencies have finished yet
            return False

    def runcase(self, case):
        super().runcase(case)
        check, partition, environ = case
        self._partitions.add(partition)

        # Set partition-based counters, if not set already
        self._running_tasks.setdefault(partition.fullname, ([], []))
        self._compiling_tasks.setdefault(partition.fullname, ([], []))
        self._ready_to_compile_tasks.setdefault(partition.fullname, ([], []))
        self._ready_to_run_tasks.setdefault(partition.fullname, ([], []))
        self._max_jobs.setdefault(partition.fullname, partition.max_jobs)

        task = RegressionTask(case, self.task_listeners)
        self._task_index[case] = task
        self.stats.add_task(task)
        self.printer.status(
            'RUN', '%s on %s using %s' %
            (check.name, partition.fullname, environ.name)
        )
        try:
            partname = partition.fullname
            if not self._setup_task(task):
                if not task.skipped and not task.failed:
                    self.printer.status(
                        'DEP', '%s on %s using %s' %
                        (check.name, partname, environ.name),
                        just='right'
                    )
                    self._waiting_tasks.append(task)

                return

            if isinstance(task.check, RunOnlyRegressionTest):
                local_index = self.local_index(task, phase='run')
            else:
                local_index = self.local_index(task, phase='compile')

            job_limit = self._rfm_max_jobs if local_index else partition.max_jobs

            def all_submissions(local, partname=None):
                if local:
                    local_tasks = 0
                    for (_, lt) in self._running_tasks.values():
                        local_tasks += len(lt)

                    for (_, lt) in self._compiling_tasks.values():
                        local_tasks += len(lt)

                    return local_tasks
                else:
                    return (
                        len(self._running_tasks[partname][local]) +
                        len(self._compiling_tasks[partname][local])
                    )

            if (all_submissions(local_index, partname) >= job_limit):
                # Make sure that we still exceeded the job limit
                getlogger().debug2(
                    f'Reached concurrency limit for partition {partname!r}: '
                    f'{partition.max_jobs} job(s)'
                )
                self._poll_tasks()

            if (all_submissions(local_index, partname) < job_limit):
                if isinstance(task.check, RunOnlyRegressionTest):
                    # Task was put in _ready_to_run_tasks during setup
                    self._ready_to_run_tasks[partname][local_index].pop()
                    self._reschedule_run(task)
                else:
                    # Task was put in _ready_to_compile_tasks during setup
                    self._ready_to_compile_tasks[partname][local_index].pop()
                    self._reschedule_compile(task)
            else:
                self.printer.status('HOLD', task.check.info(), just='right')

            # NOTE: If we don't schedule runs here and we have a lot of tests
            # compiling we will begin submitting only after all the tests are
            # processed. On the other hand I am not sure where to schedule
            # runs here.
            self._reschedule_all(phase='run')
        except TaskExit:
            if not task.failed and not task.skipped:
                with contextlib.suppress(TaskExit):
                    self._reschedule_compile(task)

            return
        except ABORT_REASONS as e:
            # If abort was caused due to failure elsewhere, abort current
            # task as well
            task.abort(e)
            self._failall(e)
            raise

    def _poll_tasks(self):
        '''Update the counts of running checks per partition.'''
        for part in self._partitions:
            partname = part.fullname
            num_tasks = len(self._running_tasks[partname][0])
            getlogger().debug2(f'Polling {num_tasks} running task(s) in '
                               f'{partname!r}')
            part_jobs = [t.check.job for t in self._running_tasks[partname][0]]
            forced_local_jobs = [t.check.job for t in self._running_tasks[partname][1]]
            part.scheduler.poll(*part_jobs)
            self.local_scheduler.poll(*forced_local_jobs)

            # Trigger notifications for finished jobs.
            # We need need a copy of the list here in order to not modify the
            # list while looping over it. `run_complete` calls `on_task_exit`,
            # which in turn will remove the task from `_running_tasks`.
            for t in self._running_tasks[partname][0] + self._running_tasks[partname][1]:
                t.run_complete()

            num_tasks = len(self._compiling_tasks[partname][0])
            getlogger().debug2(f'Polling {num_tasks} building task(s) in '
                               f'{partname!r}')
            part_jobs = [t.check.build_job for t in self._compiling_tasks[partname][0]]
            forced_local_jobs = [t.check.build_job for t in self._compiling_tasks[partname][1]]
            part.scheduler.poll(*part_jobs)
            self.local_scheduler.poll(*forced_local_jobs)

            # Trigger notifications for finished compilation jobs
            for t in self._compiling_tasks[partname][0] + self._compiling_tasks[partname][1]:
                t.compile_complete()

    def _setup_all(self):
        still_waiting = []
        for task in self._waiting_tasks:
            if (not self._setup_task(task) and
                not task.failed and not task.skipped):
                still_waiting.append(task)

        self._waiting_tasks[:] = still_waiting

    def _finalize_all(self):
        getlogger().debug2(f'Finalizing {len(self._completed_tasks)} task(s)')
        while True:
            try:
                task = self._completed_tasks.pop()
            except IndexError:
                break

            getlogger().debug2(f'Finalizing task {task.testcase}')
            with contextlib.suppress(TaskExit):
                self._finalize_task(task)

    def _finalize_task(self, task):
        getlogger().debug2(f'Finalizing task {task.testcase}')
        if not self.skip_sanity_check:
            task.sanity()

        if not self.skip_performance_check:
            task.performance()

        task.finalize()

    def _failall(self, cause):
        '''Mark all tests as failures'''
        getlogger().debug2(f'Aborting all tasks due to {type(cause).__name__}')
        for task in list(itertools.chain(*itertools.chain(*self._running_tasks.values()))):
            task.abort(cause)

        self._running_tasks = {}
        for task in list(itertools.chain(*itertools.chain(*self._compiling_tasks.values()))):
            task.abort(cause)

        self._compiling_tasks = {}
        for task in list(itertools.chain(*itertools.chain(*self._ready_to_compile_tasks.values()))):
            task.abort(cause)

        self._ready_to_compile_tasks = {}
        for task in list(itertools.chain(*itertools.chain(*self._ready_to_run_tasks.values()))):
            task.abort(cause)

        self._ready_to_run_tasks = {}
        for task in itertools.chain(self._waiting_tasks,
                                    self._completed_tasks):
            task.abort(cause)

    def _reschedule_compile(self, task):
        getlogger().debug2(f'Scheduling test case {task.testcase} for '
                           f'compiling')
        task.compile()

    def _reschedule_run(self, task):
        getlogger().debug2(f'Scheduling test case {task.testcase} for running')
        task.run()

    def _reschedule_all(self, phase='run'):
        local_tasks = 0
        for (_, lt) in self._running_tasks.values():
            local_tasks += len(lt)

        local_slots = self._rfm_max_jobs - local_tasks
        for part in self._partitions:
            partname = part.fullname
            part_tasks = (
                len(self._running_tasks[partname][0]) +
                len(self._compiling_tasks[partname][0])
            )
            part_slots = self._max_jobs[partname] - part_tasks
            num_rescheduled = 0

            for _ in range(part_slots):
                try:
                    queue = getattr(self, f'_ready_to_{phase}_tasks')
                    task = queue[partname][0].pop()
                except IndexError:
                    break

                getattr(self, f'_reschedule_{phase}')(task)
                num_rescheduled += 1

            for _ in range(local_slots):
                try:
                    queue = getattr(self, f'_ready_to_{phase}_tasks')
                    task = queue[partname][1].pop()
                except IndexError:
                    break

                getattr(self, f'_reschedule_{phase}')(task)
                local_slots -= 1
                num_rescheduled += 1

            if num_rescheduled:
                getlogger().debug2(
                    f'Rescheduled {num_rescheduled} {phase} job(s) on '
                    f'{partname!r}'
                )

    def exit(self):
        self.printer.separator('short single line',
                               'waiting for spawned checks to finish')
        while (countall(self._running_tasks) or self._waiting_tasks or
               self._completed_tasks or countall(self._ready_to_compile_tasks) or
               countall(self._compiling_tasks) or countall(self._ready_to_run_tasks)):
            getlogger().debug2(f'Running tasks: '
                               f'{countall(self._running_tasks)}')
            try:
                self._poll_tasks()

                # We count running tasks just after polling in order to check
                # more reliably that the state has changed, so that we
                # decrease the sleep time. Otherwise if the number of tasks
                # rescheduled was the as the number of tasks retired, the
                # sleep time would be increased.
                num_running = countall(self._running_tasks)
                self._finalize_all()
                self._setup_all()
                self._reschedule_all(phase='compile')
                self._reschedule_all(phase='run')
                _cleanup_all(self._retired_tasks, not self.keep_stage_files)
                if num_running:
                    self._pollctl.running_tasks(num_running).snooze()

            except TaskExit:
                with contextlib.suppress(TaskExit):
                    self._reschedule_all(phase='compile')
                    self._reschedule_all(phase='run')
            except ABORT_REASONS as e:
                self._failall(e)
                raise

        self.printer.separator('short single line',
                               'all spawned checks have finished\n')
