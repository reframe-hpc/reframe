# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import functools
import itertools
import sys
import time

from reframe.core.exceptions import (FailureLimitError,
                                     SkipTestError,
                                     TaskDependencyError,
                                     TaskExit)
from reframe.core.logging import getlogger
from reframe.frontend.executors import (ExecutionPolicy, RegressionTask,
                                        TaskEventListener, ABORT_REASONS)


def countall(d):
    return functools.reduce(lambda l, r: l + len(r), d.values(), 0)


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
        getlogger().debug2(
            f'Poll rate control: sleeping for {self._sleep_duration}s '
            f'(current poll rate: {self._num_polls/t_elapsed} polls/s)'
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

    def on_task_exit(self, task):
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

        # All currently running tasks per partition
        self._running_tasks = {}

        # Tasks that need to be finalized
        self._completed_tasks = []

        # Retired tasks that need to be cleaned up
        self._retired_tasks = []

        # Ready tasks to be executed per partition
        self._ready_tasks = {}

        # Tasks that are waiting for dependencies
        self._waiting_tasks = []

        # Job limit per partition
        self._max_jobs = {}

        # Keep a reference to all the partitions
        self._partitions = set()

        self.task_listeners.append(self)

    def _remove_from_running(self, task):
        getlogger().debug2(
            f'Removing task from the running list: {task.testcase}'
        )
        try:
            partname = task.check.current_partition.fullname
            self._running_tasks[partname].remove(task)
        except (ValueError, AttributeError, KeyError):
            getlogger().debug2('Task was not running')
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

    def on_task_setup(self, task):
        partname = task.check.current_partition.fullname
        self._ready_tasks[partname].append(task)

    def on_task_run(self, task):
        partname = task.check.current_partition.fullname
        self._running_tasks[partname].append(task)

    def on_task_skip(self, task):
        # Remove the task from the running list if it was skipped after the
        # run phase
        if task.check.current_partition:
            partname = task.check.current_partition.fullname
            if task.failed_stage in ('run_complete', 'run_wait'):
                self._running_tasks[partname].remove(task)

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
        self._running_tasks.setdefault(partition.fullname, [])
        self._ready_tasks.setdefault(partition.fullname, [])
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

            if len(self._running_tasks[partname]) >= partition.max_jobs:
                # Make sure that we still exceeded the job limit
                getlogger().debug2(
                    f'Reached concurrency limit for partition {partname!r}: '
                    f'{partition.max_jobs} job(s)'
                )
                self._poll_tasks()

            if len(self._running_tasks[partname]) < partition.max_jobs:
                # Task was put in _ready_tasks during setup
                self._ready_tasks[partname].pop()
                self._reschedule(task)
            else:
                self.printer.status('HOLD', task.check.info(), just='right')
        except TaskExit:
            if not task.failed and not task.skipped:
                with contextlib.suppress(TaskExit):
                    self._reschedule(task)

            return
        except ABORT_REASONS as e:
            # If abort was caused due to failure elsewhere, abort current
            # task as well
            task.abort(e)
            self._failall(e)
            raise

    def _poll_tasks(self):
        '''Update the counts of running checks per partition.'''

        def split_jobs(tasks):
            '''Split jobs into forced local and normal ones.'''
            forced_local = []
            normal = []
            for t in tasks:
                if t.check.local:
                    forced_local.append(t.check.job)
                else:
                    normal.append(t.check.job)

            return forced_local, normal

        for part in self._partitions:
            partname = part.fullname
            num_tasks = len(self._running_tasks[partname])
            getlogger().debug2(f'Polling {num_tasks} task(s) in {partname!r}')
            forced_local_jobs, part_jobs = split_jobs(
                self._running_tasks[partname]
            )
            part.scheduler.poll(*part_jobs)
            self.local_scheduler.poll(*forced_local_jobs)

            # Trigger notifications for finished jobs
            for t in self._running_tasks[partname]:
                t.run_complete()

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
        for task in list(itertools.chain(*self._running_tasks.values())):
            task.abort(cause)

        self._running_tasks = {}
        for ready_list in self._ready_tasks.values():
            for task in ready_list:
                task.abort(cause)

        for task in itertools.chain(self._waiting_tasks,
                                    self._completed_tasks):
            task.abort(cause)

    def _reschedule(self, task):
        getlogger().debug2(f'Scheduling test case {task.testcase} for running')
        task.compile()
        task.compile_wait()
        task.run()

    def _reschedule_all(self):
        for partname, tasks in self._running_tasks.items():
            num_tasks = len(tasks)
            num_empty_slots = self._max_jobs[partname] - num_tasks
            num_rescheduled = 0
            for _ in range(num_empty_slots):
                try:
                    task = self._ready_tasks[partname].pop()
                except IndexError:
                    break

                self._reschedule(task)
                num_rescheduled += 1

            if num_rescheduled:
                getlogger().debug2(
                    f'Rescheduled {num_rescheduled} job(s) on {partname!r}'
                )

    def exit(self):
        self.printer.separator('short single line',
                               'waiting for spawned checks to finish')
        while (countall(self._running_tasks) or self._waiting_tasks or
               self._completed_tasks or countall(self._ready_tasks)):
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
                self._reschedule_all()
                _cleanup_all(self._retired_tasks, not self.keep_stage_files)
                if num_running:
                    self._pollctl.running_tasks(num_running).snooze()

            except TaskExit:
                with contextlib.suppress(TaskExit):
                    self._reschedule_all()
            except ABORT_REASONS as e:
                self._failall(e)
                raise

        self.printer.separator('short single line',
                               'all spawned checks have finished\n')
