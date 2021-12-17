# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
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
            'RUN',
            f'{check.name} on {partition.fullname} using {environ.name}'
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


######################### Stages of the pipeline #########################
#
# Each test starts from the `wait` stage and in the last step of the
# policy there a loop where all tests are bumped to the next phase
# if possible.
#
#           +--------------------[ wait ]
#           |                        |
#           |               if all deps finished and
#           |                 test is not RunOnly
#           |                        |
#           |                        ↓
#           |               [ ready_to_compile ]
#           |                        |
#           |              if there are available
#           |                      slots
#  if all deps finished and          |
#     test is RunOnly                ↓
#           |                  [ compiling ]--------------+
#           |                        |                    |
#           |         if compilation has finished and     |
#           |            test is not CompileOnly          |
#           |                        |                    |
#           |                        ↓                    |
#           +--------------->[ ready_to_run ]             |
#                                    |                    |
#                          if there are available         |
#                                  slots                  |
#                                    |      if compilation has finished and
#                                    ↓           test is CompileOnly
#                               [ running ]               |
#                                    |                    |
#                          if job has finished            |
#   tests can exit the               |                    |
#  pipeline at any point             ↓                    |
#     if they fail             [ completed ]<-------------+
#          :                         |
#          :              if sanity and performance
#          |                      succeed
#          |                         |
#          ↓                         ↓
#      ( failed )               ( retired )

class AsynchronousExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):
        super().__init__()

        self._pollctl = _PollController()

        # Index tasks by test cases
        self._task_index = {}

        # A set of all the current tasks
        self._current_tasks = set()

        # Keep a reference to all the partitions
        self._partitions = set()

        # Sets of the jobs that should be polled for each partition
        self._scheduler_tasks = {
            '_rfm_local': set()
        }

        # Retired tasks that need to be cleaned up
        self._retired_tasks = []

        # Job limit per partition
        self._max_jobs = {
            '_rfm_local': rt.runtime().get_option(f'systems/0/rfm_max_jobs')
        }

        self.task_listeners.append(self)

    def runcase(self, case):
        super().runcase(case)
        check, partition, environ = case
        self._partitions.add(partition)

        # Set partition-based counters, if not set already
        self._scheduler_tasks.setdefault(partition.fullname, set())
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
        while self._current_tasks:
            try:
                self._poll_tasks()
                num_running = sum(
                    1 if t.policy_stage in ('running', 'compiling') else 0
                    for t in self._current_tasks
                )
                self.advance_all(self._current_tasks)
                _cleanup_all(self._retired_tasks, not self.keep_stage_files)
                if num_running:
                    self._pollctl.running_tasks(num_running).snooze()
            except ABORT_REASONS as e:
                self._failall(e)
                raise

    def _poll_tasks(self):
        for part in self._partitions:
            jobs = []
            for t in self._scheduler_tasks[part.fullname]:
                if t.policy_stage == 'compiling':
                    jobs.append(t.check.build_job)
                elif t.policy_stage == 'running':
                    jobs.append(t.check.job)

            part.scheduler.poll(*jobs)

        jobs = []
        for t in self._scheduler_tasks['_rfm_local']:
            if t.policy_stage == 'compiling':
                jobs.append(t.check.build_job)
            elif t.policy_stage == 'running':
                jobs.append(t.check.job)

        self.local_scheduler.poll(*jobs)

    def advance_all(self, tasks, timeout=None):
        t_init = time.time()
        num_progressed = 0

        getlogger().debug2(f"Current tests: {len(tasks)}")
        # progress might remove the tasks that retire or fail
        for t in list(tasks):
            bump_state = getattr(self, f'advance_{t.policy_stage}')
            num_progressed += bump_state(t)
            t_elapsed = time.time() - t_init
            if timeout and t_elapsed > timeout and num_progressed:
                break

        getlogger().debug2(f"Bumped {num_progressed} test(s).")

    def advance_wait(self, task):
        if self.deps_skipped(task):
            try:
                raise SkipTestError('skipped due to skipped dependencies')
            except SkipTestError as e:
                task.skip()
                self._current_tasks.remove(task)
                return 1
        elif self.deps_succeeded(task):
            try:
                self.printer.status(
                    'RUN', f'{task.check.name} on '
                    f'{task.testcase.partition.fullname} using '
                    f'{task.testcase.environ.name}'
                )
                task.setup(task.testcase.partition,
                           task.testcase.environ,
                           sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                           sched_options=self.sched_options)
            except TaskExit:
                self._current_tasks.remove(task)
                return 1

            if isinstance(task.check, RunOnlyRegressionTest):
                try:
                    task.compile()
                    task.compile_wait()
                except TaskExit:
                    # Run and run_wait are no-ops for
                    # CompileOnlyRegressionTest. This shouldn't fail.
                    self._current_tasks.remove(task)
                    return 1

                task.policy_stage = 'ready_to_run'
            else:
                task.policy_stage = 'ready_to_compile'

            return 1
        elif self.deps_failed(task):
            exc = TaskDependencyError('dependencies failed')
            task.fail((type(exc), exc, None))
            self._current_tasks.remove(task)
            return 1
        else:
            # Not all dependencies have finished yet
            return 0

    def advance_ready_to_compile(self, task):
        partname = (
            '_rfm_local' if task.check.local or task.check.build_locally
            else task.check.current_partition.fullname
        )
        if len(self._scheduler_tasks[partname]) < self._max_jobs[partname]:
            try:
                task.compile()
                task.policy_stage = 'compiling'
                self._scheduler_tasks[partname].add(task)
            except TaskExit:
                self._current_tasks.remove(task)

            return 1

        return 0

    def advance_compiling(self, task):
        partname = (
            '_rfm_local' if task.check.local or task.check.build_locally
            else task.check.current_partition.fullname
        )
        try:
            if task.compile_complete():
                task.compile_wait()
                self._scheduler_tasks[partname].remove(task)

                if isinstance(task.check, CompileOnlyRegressionTest):
                    try:
                        task.run()
                        task.run_wait()
                    except TaskExit:
                        # Run and run_wait are no-ops for
                        # CompileOnlyRegressionTest. This shouldn't fail.
                        self._current_tasks.remove(task)
                        return 1

                    task.policy_stage = 'completed'
                else:
                    task.policy_stage = 'ready_to_run'

                return 1
            else:
                return 0

        except TaskExit:
            self._scheduler_tasks[partname].remove(task)
            self._current_tasks.remove(task)
            return 1

    def advance_ready_to_run(self, task):
        partname = (
            '_rfm_local' if task.check.local
            else task.check.current_partition.fullname
        )
        if len(self._scheduler_tasks[partname]) < self._max_jobs[partname]:
            try:
                task.run()
                task.policy_stage = 'running'
                self._scheduler_tasks[partname].add(task)
            except TaskExit:
                self._current_tasks.remove(task)

            return 1

        return 0

    def advance_running(self, task):
        partname = (
            '_rfm_local' if task.check.local
            else task.check.current_partition.fullname
        )
        try:
            if task.run_complete():
                task.run_wait()
                self._scheduler_tasks[partname].remove(task)

                task.policy_stage = 'completed'
                return 1
            else:
                return 0

        except TaskExit:
            self._scheduler_tasks[partname].remove(task)
            self._current_tasks.remove(task)
            return 1

    def advance_completed(self, task):
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

    # TODO all this prints have to obviously leave from here...
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
        pass

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

        for c in task.testcase.deps:
            # NOTE: Restored dependencies are not in the task_index
            if c in self._task_index:
                self._task_index[c].ref_count -= 1
