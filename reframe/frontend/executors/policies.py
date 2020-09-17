# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import functools
import itertools
import math
import sys
import time

from datetime import datetime

from reframe.core.exceptions import (TaskDependencyError, TaskExit)
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


def init_sleep_control(num_tasks):
    num_prev_tasks = num_tasks
    sleep_min = 0.5
    sleep_max = 10
    sleep_inc = 0.5
    sleep_next = sleep_min

    def _sleep_duration(num_tasks):
        nonlocal sleep_next, num_prev_tasks

        if num_tasks != num_prev_tasks or sleep_next >= sleep_max:
            sleep_next = sleep_min
            num_prev_tasks = num_tasks
        else:
            sleep_next += sleep_inc

        return sleep_next

    return _sleep_duration


class SerialExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):
        super().__init__()

        # Index tasks by test cases
        self._task_index = {}

        # All schedulers per partition
        self._schedulers = {}

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
            if any(self._task_index[c].failed for c in case.deps):
                raise TaskDependencyError('dependencies failed')

            partname = task.testcase.partition.fullname
            sched = self._schedulers.get(partname)
            task.setup(task.testcase.partition,
                       task.testcase.environ,
                       scheduler=sched,
                       sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                       sched_account=self.sched_account,
                       sched_partition=self.sched_partition,
                       sched_reservation=self.sched_reservation,
                       sched_nodelist=self.sched_nodelist,
                       sched_exclude_nodelist=self.sched_exclude_nodelist,
                       sched_options=self.sched_options)

            self._schedulers.setdefault(partname, task.check.job.scheduler)
            sched = task.check.job.scheduler

            task.compile()
            task.compile_wait()
            task.run()
            nap_duration = init_sleep_control(1)
            while True:
                sched.poll(task.check.job)
                if task.poll():
                    break

                t = nap_duration(int(not task.completed))
                getlogger().debug(f'sleeping for {t}s')
                time.sleep(t)

            task.wait()
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

    def on_task_failure(self, task):
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
        getlogger().verbose(f"==> {timings}")

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
        getlogger().verbose(f"==> {timings}")
        # update reference count of dependencies
        for c in task.testcase.deps:
            self._task_index[c].ref_count -= 1

        _cleanup_all(self._retired_tasks, not self.keep_stage_files)

    def exit(self):
        # Clean up all remaining tasks
        _cleanup_all(self._retired_tasks, not self.keep_stage_files)


class AsynchronousExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):

        super().__init__()

        # Index tasks by test cases
        self._task_index = {}

        # All currently running tasks per partition
        self._running_tasks = {}

        # All schedulers per partition
        self._schedulers = {}

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

        self.task_listeners.append(self)

    def _remove_from_running(self, task):
        getlogger().debug(
            'removing task from running list: %s' % task.check.info()
        )
        try:
            partname = task.check.current_partition.fullname
            self._running_tasks[partname].remove(task)
        except (ValueError, AttributeError, KeyError):
            getlogger().debug('not in running tasks')
            pass

    def deps_failed(self, task):
        return any(self._task_index[c].failed for c in task.testcase.deps)

    def deps_succeeded(self, task):
        return all(self._task_index[c].succeeded for c in task.testcase.deps)

    def on_task_setup(self, task):
        partname = task.check.current_partition.fullname
        self._ready_tasks[partname].append(task)
        self._schedulers.setdefault(partname, task.check.job.scheduler)

    def on_task_run(self, task):
        partname = task.check.current_partition.fullname
        self._running_tasks[partname].append(task)

    def on_task_failure(self, task):
        msg = f'{task.check.info()} [{task.pipeline_timings_basic()}]'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self._remove_from_running(task)
            self.printer.status('FAIL', msg, just='right')

        getlogger().verbose(f"==> {task.pipeline_timings_all()}")

    def on_task_success(self, task):
        msg = f'{task.check.info()} [{task.pipeline_timings_basic()}]'
        self.printer.status('OK', msg, just='right')
        getlogger().verbose(f"==> {task.pipeline_timings_all()}")
        # update reference count of dependencies
        for c in task.testcase.deps:
            self._task_index[c].ref_count -= 1

        self._retired_tasks.append(task)

    def on_task_exit(self, task):
        task.wait()
        self._remove_from_running(task)
        self._completed_tasks.append(task)

    def _setup_task(self, task):
        if self.deps_succeeded(task):
            try:
                sched = self._schedulers.get(task.testcase.partition.fullname,
                                             None)
                task.setup(task.testcase.partition,
                           task.testcase.environ,
                           scheduler=sched,
                           sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                           sched_account=self.sched_account,
                           sched_partition=self.sched_partition,
                           sched_reservation=self.sched_reservation,
                           sched_nodelist=self.sched_nodelist,
                           sched_exclude_nodelist=self.sched_exclude_nodelist,
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
                if not task.failed:
                    self.printer.status(
                        'DEP', '%s on %s using %s' %
                        (check.name, partname, environ.name),
                        just='right'
                    )
                    self._waiting_tasks.append(task)

                return

            if len(self._running_tasks[partname]) >= partition.max_jobs:
                # Make sure that we still exceeded the job limit
                getlogger().debug('reached job limit (%s) for partition %s' %
                                  (partition.max_jobs, partname))
                self._poll_tasks()

            if len(self._running_tasks[partname]) < partition.max_jobs:
                # Task was put in _ready_tasks during setup
                self._ready_tasks[partname].pop()
                self._reschedule(task)
            else:
                self.printer.status('HOLD', task.check.info(), just='right')
        except TaskExit:
            if not task.failed:
                with contextlib.suppress(TaskExit):
                    self._reschedule(task)

            return
        except ABORT_REASONS as e:
            if not task.failed:
                # Abort was caused due to failure elsewhere, abort current
                # task as well
                task.abort(e)

            self._failall(e)
            raise

    def _poll_tasks(self):
        '''Update the counts of running checks per partition.'''
        getlogger().debug('updating counts for running test cases')
        for partname, sched in self._schedulers.items():
            getlogger().debug(f'polling {len(self._running_tasks[partname])} '
                              f'task(s) in {partname}')
            sched.poll(
                *(task.check.job for task in self._running_tasks[partname])
            )

            for t in self._running_tasks[partname]:
                t.poll()

    def _setup_all(self):
        still_waiting = []
        for task in self._waiting_tasks:
            if not self._setup_task(task) and not task.failed:
                still_waiting.append(task)

        self._waiting_tasks[:] = still_waiting

    def _finalize_all(self):
        getlogger().debug('finalizing tasks: %s', len(self._completed_tasks))
        while True:
            try:
                task = self._completed_tasks.pop()
            except IndexError:
                break

            getlogger().debug('finalizing task: %s' % task.check.info())
            with contextlib.suppress(TaskExit):
                self._finalize_task(task)

    def _finalize_task(self, task):
        if not self.skip_sanity_check:
            task.sanity()

        if not self.skip_performance_check:
            task.performance()

        task.finalize()

    def _failall(self, cause):
        '''Mark all tests as failures'''
        for task in list(itertools.chain(*self._running_tasks.values())):
            task.abort(cause)

        self._running_tasks = {}
        for ready_list in self._ready_tasks.values():
            getlogger().debug('ready list size: %s' % len(ready_list))
            for task in ready_list:
                task.abort(cause)

        for task in itertools.chain(self._waiting_tasks,
                                    self._retired_tasks,
                                    self._completed_tasks):
            task.abort(cause)

    def _reschedule(self, task):
        getlogger().debug('scheduling test case for running')

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
                getlogger().debug('rescheduled %s job(s) on %s' %
                                  (num_rescheduled, partname))

    def exit(self):
        self.printer.separator('short single line',
                               'waiting for spawned checks to finish')
        nap_duration = init_sleep_control(countall(self._running_tasks))
        while (countall(self._running_tasks) or self._waiting_tasks or
               self._completed_tasks or countall(self._ready_tasks)):
            getlogger().debug(f'running tasks: '
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
                t = nap_duration(num_running)
                getlogger().debug(f'sleeping for {t}s')
                time.sleep(t)

            except TaskExit:
                with contextlib.suppress(TaskExit):
                    self._reschedule_all()
            except ABORT_REASONS as e:
                self._failall(e)
                raise

        self.printer.separator('short single line',
                               'all spawned checks have finished\n')
