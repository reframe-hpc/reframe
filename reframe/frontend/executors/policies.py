# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import contextlib
import math
import signal
import sys
import time

import reframe.core.runtime as rt
import reframe.utility as util
from reframe.core.exceptions import (FailureLimitError,
                                     RunSessionTimeout,
                                     SkipTestError,
                                     TaskDependencyError,
                                     TaskExit,
                                     KeyboardError)
from reframe.core.logging import getlogger, level_from_str
from reframe.frontend.executors import (ExecutionPolicy, RegressionTask,
                                        TaskEventListener, ABORT_REASONS)
from reframe.frontend.executors import asyncio_run
from reframe.utility.osext import current_task, all_tasks

def _get_partition_name(task, phase='run'):
    if (task.check.local or
        (phase == 'build' and task.check.build_locally)):
        return '_rfm_local'
    else:
        return task.check.current_partition.fullname


async def _cleanup_all(tasks, *args, **kwargs):
    for task in tasks:
        if task.ref_count == 0:
            with contextlib.suppress(TaskExit):
                await task.cleanup(*args, **kwargs)

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
        self._sleep_duration = {}
        self._t_init = {}
        self._t_init_polling = {}
        self._t_snoozed = {}
        self._jobs_to_pool = {}
        self._jobs_pooling = {}
        self._poll_event = {}

    def reset_snooze_time(self, sched):
        if self._sleep_duration.get(sched) is None:
            self._sleep_duration[sched] = self.SLEEP_MIN
        if self._t_init_polling.get(sched) is None:
            self._t_init_polling[sched] = time.time()
        if self._poll_event.get(sched) is None:
            self._poll_event[sched] = asyncio.Event()
            self._poll_event[sched].set()
        if self._jobs_pooling.get(sched) is None:
            self._jobs_pooling[sched] = []

    async def snooze(self, sched):
        if self._num_polls == 0:
            self._t_init[sched] = time.time()
            self._t_snoozed[sched] = time.time()

        t_elapsed = time.time() - self._t_init_polling[sched]
        poll_rate = self._num_polls / t_elapsed if t_elapsed else math.inf
        getlogger().debug2(
            f'Poll rate control: sleeping for {self._sleep_duration[sched]}s '
            f'(current poll rate: {poll_rate} polls/s)'
        )
        await asyncio.sleep(self._sleep_duration[sched])

    def is_time_to_poll(self, sched):
        # We check here if it's time to poll
        if self._t_init.get(sched) is None:
            self._t_init[sched] = time.time()
            self._t_snoozed[sched] = time.time()
            self._num_polls += 1
            return True

        t_elapsed = time.time() - self._t_init[sched]
        getlogger().debug(f'Time since last poll: {t_elapsed}, {self._sleep_duration[sched]}')

        if t_elapsed >= self._sleep_duration[sched]:
            self._num_polls += 1
            return True
        else:
            return False

    def reset_time_to_poll(self, sched):
        t_increase_sleep = time.time() - self._t_snoozed[sched]
        if t_increase_sleep > self._sleep_duration[sched]:
            self._t_snoozed[sched] = time.time()
            self._sleep_duration[sched] = min(
                self._sleep_duration[sched]*self.SLEEP_INC_RATE, self.SLEEP_MAX
            )
        self._t_init[sched] = time.time()
        self._t_snoozed[sched] = time.time()


global _poll_controller
_poll_controller = _PollController()


def getpollcontroller():
    return _poll_controller


class SerialExecutionPolicy(ExecutionPolicy, TaskEventListener):
    def __init__(self):
        super().__init__()

        self._pollctl = _PollController()

        # Index tasks by test cases
        self._task_index = {}

        # Tasks that have finished, but have not performed their cleanup phase
        self._retired_tasks = []
        self.task_listeners.append(self)

    async def _runcase(self, case, task):
        check, partition, _ = case
        if check.is_dry_run():
            self.printer.status('DRY', task.info())
        else:
            self.printer.status('RUN', task.info())

        self._task_index[case] = task
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

            await task.setup(task.testcase.partition,
                       task.testcase.environ,
                       sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                       sched_options=self.sched_options)
            await task.compile()
            await task.compile_wait()
            await task.run()

            # Pick the right scheduler
            if task.check.local:
                sched = self.local_scheduler
            else:
                sched = partition.scheduler

            self._pollctl.reset_snooze_time(sched.registered_name)
            while True:
                if not self.dry_run_mode:
                    await sched.poll(task.check.job)
                if task.run_complete():
                    break

                await self._pollctl.snooze(sched.registered_name)
            await task.run_wait()
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
            if type(e) is KeyboardInterrupt:
                raise KeyboardError
            else:
                raise e
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

    def on_task_abort(self, task):
        msg = f'{task.info()}'
        self.printer.status('ABORT', msg, just='right')

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

        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

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

        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def execute(self, testcases):
        '''Execute the policy for a given set of testcases.'''
        # Moved here the execution
        try:
            loop = asyncio.get_event_loop()
            for task in all_tasks(loop):
                if isinstance(task, asyncio.tasks.Task):
                    try:
                        task.cancel()
                    except RuntimeError:
                        pass
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                watcher = asyncio.get_child_watcher()
                if isinstance(watcher, asyncio.SafeChildWatcher):
                    # Detach the watcher from the current loop to avoid issues
                    watcher.close()
                    watcher.attach_loop(None)
                asyncio.set_event_loop(loop)
                if isinstance(watcher, asyncio.SafeChildWatcher):
                    # Reattach the watcher to the new loop
                    watcher.attach_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for case in testcases:
            try:
                task = RegressionTask(case, self.task_listeners)
                self.stats.add_task(task)
                loop.run_until_complete(self._runcase(case, task))
            except (Exception, KeyboardInterrupt) as e:
                if type(e) in ABORT_REASONS:
                    # When the KeyboardInterrupt happens while asyncio.sleep it comes here
                    if not task.aborted:
                        # Make sure that the task is aborted in that case
                        task.abort(e)
                    for task in all_tasks(loop):
                        if isinstance(task, asyncio.tasks.Task):
                            task.cancel()
                    loop.close()
                    # In case we still receive the KeyboardInterrupt because it happened inside
                    # asyncio.sleep()
                    if isinstance(e, KeyboardInterrupt):
                        raise KeyboardError
                    else:
                        raise e
                else:
                    getlogger().info(f"Execution stopped due to an error: {e}")
                    break
        loop.close()
        self._exit()

    def _exit(self):
        pass
        # Clean up all remaining tasks
        # asyncio_run(_cleanup_all(self._retired_tasks, not self.keep_stage_files))


class AsyncioExecutionPolicy(ExecutionPolicy, TaskEventListener):
    '''The asynchronous execution policy.'''

    def __init__(self):
        super().__init__()

        self._pollctl = getpollcontroller()
        self._current_tasks = util.OrderedSet()

        # Index tasks by test cases
        self._task_index = {}

        # Tasks per partition
        self._partition_tasks = {
            '_rfm_local': util.OrderedSet()
        }

        # Job limit per partition
        self._max_jobs = {
            '_rfm_local': rt.runtime().get_option('systems/0/max_local_jobs')
        }
        self._pipeline_statistics = rt.runtime().get_option(
            'general/0/dump_pipeline_progress'
        )
        # Tasks that have finished, but have not performed their cleanup phase
        self._retired_tasks = []
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

    async def _runcase(self, case, task):
        # I added the task here as an argument because,
        # I wanted to initialize it
        # outside, when I gather the tasks.
        # If I gather the tasks and then I do asyncio
        # if one of them fails the others are not iformed,
        # I had to code that manually. There is a way to make everything
        # stop if an exepction is raised but I didn't know how to treat
        # that raise Exception nicelly because I wouldn't be able
        # to abort the tasks which the execution has not yet started,
        # I needed to do abortall on all the tests, not only the ones
        # which were initiated by the execution. Exit gracefully
        # the execuion loop aborting all the tasks
        try:
            check, partition, _ = case

            self._partition_tasks.setdefault(partition.fullname, util.OrderedSet())
            self._max_jobs.setdefault(partition.fullname, partition.max_jobs)

            self._task_index[case] = task

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

            deps_status = await self.check_deps(task)
            if deps_status == "skipped":
                try:
                    raise SkipTestError('skipped due to skipped dependencies')
                except SkipTestError:
                    task.skip()
                    self._current_tasks.remove(task)
                    return 1
            elif deps_status == "succeeded":
                if task.check.is_dry_run():
                    self.printer.status('DRY', task.info())
                else:
                    self.printer.status('RUN', task.info())
            elif deps_status == "failed":
                exc = TaskDependencyError('dependencies failed')
                task.fail((type(exc), exc, None))
                self._current_tasks.remove(task)
                return 1

            await task.setup(task.testcase.partition,
                             task.testcase.environ,
                             sched_flex_alloc_nodes=self.sched_flex_alloc_nodes,
                             sched_options=self.sched_options)

            partname = _get_partition_name(task, phase='build')
            max_jobs = self._max_jobs[partname]
            while len(self._partition_tasks[partname])+1 > max_jobs:
                getlogger().debug2(f'Hit the max job limit of {partname}: {max_jobs}')
                await asyncio.sleep(0.001)
            self._partition_tasks[partname].add(task)
            await task.compile()

            # If RunOnly, no polling for run jobs
            if task.check.build_job:
                # Pick the right scheduler
                if task.check.build_locally:
                    sched = self.local_scheduler
                else:
                    sched = partition.scheduler

                self._pollctl.reset_snooze_time(sched.registered_name)
                while True:
                    if not self.dry_run_mode:
                        if (getpollcontroller().is_time_to_poll(sched.registered_name)): # and
                            # getpollcontroller()._poll_event[sched.registered_name].is_set()):
                            getlogger().debug2("Jobs to poll"
                                               f"{set(getpollcontroller()._jobs_to_pool[sched.registered_name])}")
                            getlogger().debug2("Jobs polling"
                                               f"{set(getpollcontroller()._jobs_pooling[sched.registered_name])}")
                            jobs_pool = list(set(getpollcontroller()._jobs_to_pool[
                                                sched.registered_name
                                                ]) - set(getpollcontroller()._jobs_pooling[
                                                sched.registered_name
                                                ]))
                            getlogger().debug2(f"jobs poll {jobs_pool}")
                            if jobs_pool:
                                getpollcontroller()._jobs_pooling[sched.registered_name] += jobs_pool
                                getlogger().debug2("Polling, jobs polling"
                                                   f"{set(getpollcontroller()._jobs_pooling[sched.registered_name])}")
                                getpollcontroller().reset_time_to_poll(sched.registered_name)
                                # getpollcontroller()._poll_event[sched.registered_name].clear()
                                await sched.poll(*jobs_pool)
                                # Check if reframe was aborted but we were waiting for the command
                                if hasattr(current_task(), 'aborting'):
                                    raise asyncio.CancelledError
                                getpollcontroller()._jobs_pooling[sched.registered_name] = [
                                    job for job in getpollcontroller()._jobs_pooling[sched.registered_name]
                                    if job not in jobs_pool
                                ]
                                getlogger().debug2("After polling, jobs polling"
                                                   f"{set(getpollcontroller()._jobs_pooling[sched.registered_name])}")
                                # await asyncio.sleep(0)
                                # getpollcontroller()._poll_event[sched.registered_name].set()

                    if task.compile_complete():
                        break
                    await self._pollctl.snooze(sched.registered_name)
                    if task.compile_complete():
                        break
                    # else:
                    #     # yield control to another task to give it the chance to check their status
                    #     await asyncio.sleep(0)
                    # We need to check the timeout inside the while loop
                    if self.timeout_expired():
                        raise RunSessionTimeout(
                            'maximum session duration exceeded'
                        )
            else:
                if self._pipeline_statistics:
                    getlogger().debug2(f"{task}, compiling to ready_run")
                    self._update_pipeline_progress('compiling', 'ready_run', 1)
            await task.compile_wait()
            self._partition_tasks[partname].remove(task)
            partname = _get_partition_name(task, phase='run')
            max_jobs = self._max_jobs[partname]
            while len(self._partition_tasks[partname])+1 > max_jobs:
                await asyncio.sleep(0.001)
            self._partition_tasks[partname].add(task)
            await task.run()
            # If CompileOnly, no polling for run jobs
            if task.check.job:
                # Pick the right scheduler
                if task.check.local:
                    sched = self.local_scheduler
                else:
                    sched = partition.scheduler

                self._pollctl.reset_snooze_time(sched.registered_name)
                while True:
                    if not self.dry_run_mode:
                        if (getpollcontroller().is_time_to_poll(sched.registered_name)): # and
                            # getpollcontroller()._poll_event[sched.registered_name].is_set()):
                            getlogger().debug2("Jobs to poll"
                                               f"{set(getpollcontroller()._jobs_to_pool[sched.registered_name])}")
                            getlogger().debug2("Jobs polling"
                                               f"{set(getpollcontroller()._jobs_pooling[sched.registered_name])}")
                            jobs_pool = list(set(getpollcontroller()._jobs_to_pool[
                                                sched.registered_name
                                                ]) - set(getpollcontroller()._jobs_pooling[
                                                sched.registered_name
                                                ]))
                            getlogger().debug2(f"jobs poll {jobs_pool}")
                            if jobs_pool:
                                getpollcontroller()._jobs_pooling[sched.registered_name] += jobs_pool
                                getlogger().debug2("Polling, jobs polling"
                                                   f"{set(getpollcontroller()._jobs_pooling[sched.registered_name])}")
                                getpollcontroller().reset_time_to_poll(sched.registered_name)
                                # getpollcontroller()._poll_event[sched.registered_name].clear()
                                await sched.poll(*jobs_pool)
                                # Check if reframe was aborted but we were waiting for the command
                                if hasattr(current_task(), 'aborting'):
                                    raise asyncio.CancelledError
                                getpollcontroller()._jobs_pooling[sched.registered_name] = [
                                    job for job in getpollcontroller()._jobs_pooling[sched.registered_name]
                                    if job not in jobs_pool
                                ]
                                getlogger().debug2("After polling, jobs polling"
                                                   f"{set(getpollcontroller()._jobs_pooling[sched.registered_name])}")
                                # getpollcontroller()._poll_event[sched.registered_name].set()

                    if task.run_complete():
                        break
                    await self._pollctl.snooze(sched.registered_name)
                    if task.run_complete():
                        break
                    if self.timeout_expired():
                        raise RunSessionTimeout(
                            'maximum session duration exceeded'
                        )
            else:
                if self._pipeline_statistics:
                    getlogger().debug2(f"{task}, running to completing")
                    self._update_pipeline_progress('running', 'completing', 1)
            await task.run_wait()
            self._partition_tasks[partname].remove(task)

            if not self.skip_sanity_check:
                task.sanity()

            if not self.skip_performance_check:
                task.performance()

            task.finalize()
            self._retired_tasks.append(task)

            if self._pipeline_statistics:
                self._update_pipeline_progress('completing', 'retired', 1)

            # await _cleanup_all(self._retired_tasks, not self.keep_stage_files)
            if task.ref_count == 0:
                with contextlib.suppress(TaskExit):
                    await task.cleanup(not self.keep_stage_files)

            if self._pipeline_statistics:
                self._update_pipeline_progress(
                    'retired', 'completed', 1
                )

        except TaskExit:
            self._current_tasks.remove(task)
            if task.check.current_partition:
                partname = task.check.current_partition.fullname
            else:
                partname = None

            # Remove tasks from the partition tasks if there
            with contextlib.suppress(KeyError):
                self._partition_tasks['_rfm_local'].remove(task)
            with contextlib.suppress(KeyError):
                if partname:
                    self._partition_tasks[partname].remove(task)

            return
        except ABORT_REASONS as e:
            task.abort()
            raise
        except asyncio.CancelledError:
            task.abort()
        except BaseException:
            task.fail(sys.exc_info())
            self._current_tasks.remove(task)
            if task.check.current_partition:
                partname = task.check.current_partition.fullname
            else:
                partname = None

            # Remove tasks from the partition tasks if there
            with contextlib.suppress(KeyError):
                self._partition_tasks['_rfm_local'].remove(task)
            with contextlib.suppress(KeyError):
                if partname:
                    self._partition_tasks[partname].remove(task)
            return

    async def check_deps(self, task):
        while not (self.deps_skipped(task) or self.deps_failed(task) or
                   self.deps_succeeded(task)):
            getlogger().debug2(f'{task.info()} waiting for dependencies')
            await asyncio.sleep(0.001)

        if self.deps_skipped(task):
            return "skipped"
        elif self.deps_failed(task):
            return "failed"
        elif self.deps_succeeded(task):
            return "succeeded"

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
        if self._pipeline_statistics:
            self._update_pipeline_progress('startup', 'ready_compile', 1)

    def on_task_run(self, task):
        if task.check.job:
            if getpollcontroller()._jobs_to_pool.get(
                task.check.job.scheduler.registered_name
            ):
                getpollcontroller()._jobs_to_pool[
                    task.check.job.scheduler.registered_name
                ].append(task.check.job)
            else:
                getpollcontroller()._jobs_to_pool[
                    task.check.job.scheduler.registered_name
                ] = [task.check.job]
        if self._pipeline_statistics:
            self._update_pipeline_progress('ready_run', 'running', 1)

    def on_task_compile(self, task):
        if task.check.build_job:
            if getpollcontroller()._jobs_to_pool.get(
                task.check.build_job.scheduler.registered_name
            ):
                getpollcontroller()._jobs_to_pool[
                    task.check.build_job.scheduler.registered_name
                ].append(task.check.build_job)
            else:
                getpollcontroller()._jobs_to_pool[
                    task.check.build_job.scheduler.registered_name
                ] = [task.check.build_job]
        if self._pipeline_statistics:
            self._update_pipeline_progress('ready_compile', 'compiling', 1)

    def on_task_exit(self, task):
        if task.check.job:
            getpollcontroller()._jobs_to_pool[
                task.check.job.scheduler.registered_name
            ].remove(task.check.job)
            getpollcontroller().reset_snooze_time(task.check.job.scheduler.registered_name)
        if self._pipeline_statistics:
            self._update_pipeline_progress('running', 'completing', 1)

    def on_task_compile_exit(self, task):
        if task.check.build_job:
            getpollcontroller().reset_snooze_time(task.check.build_job.scheduler.registered_name)
            getpollcontroller()._jobs_to_pool[
                task.check.build_job.scheduler.registered_name
            ].remove(
                task.check.build_job)
        if self._pipeline_statistics:
            self._update_pipeline_progress('compiling', 'ready_run', 1)

    def on_task_skip(self, task):
        msg = str(task.exc_info[1])
        self.printer.status('SKIP', msg, just='right')
        if self._pipeline_statistics:
            self._update_pipeline_progress('startup', 'skip', 1)

    def on_task_abort(self, task):
        msg = f'{task.info()}'
        self.printer.status('ABORT', msg, just='right')

    def on_task_failure(self, task):
        self._num_failed_tasks += 1
        msg = f'{task.info()}'
        if task.failed_stage == 'cleanup':
            self.printer.status('ERROR', msg, just='right')
        else:
            self.printer.status('FAIL', msg, just='right')

        if self._pipeline_statistics:
            old_state = task._failed_state
            if old_state == 'running':
                old_state = 'ready_run'
            elif old_state == 'compiling':
                old_state = 'ready_compile'
            self._update_pipeline_progress(old_state, 'fail', 1)

        _print_perf(task)
        if task.failed_stage == 'sanity':
            # Dry-run the performance stage to trigger performance logging
            task.performance(dry_run=True)

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

        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

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

        # _cleanup_all(self._retired_tasks, not self.keep_stage_files)
        if self.timeout_expired():
            raise RunSessionTimeout('maximum session duration exceeded')

    def _exit(self):
        # Clean up all remaining tasks
        asyncio_run(_cleanup_all(self._retired_tasks, not self.keep_stage_files))
        if self._pipeline_statistics:
            self._dump_pipeline_progress('pipeline-progress.json')

    def execute(self, testcases):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        all_cases = []
        for t in testcases:
            task = RegressionTask(t, self.task_listeners)
            self.stats.add_task(task)
            # Add the tasks outside the asyncio handling so that all tasks are aborted
            # otherwise the task in the TesStats is not updated accordingly
            self._current_tasks.add(task)
            all_cases.append(asyncio.ensure_future(self._runcase(t, task)))
        if self._pipeline_statistics:
            self._init_pipeline_progress(len(self._current_tasks))
        try:
            # Wait for tasks until the first failure
            # loop.add_signal_handler(signal.SIGINT, lambda:  asyncio.ensure_future(handle_sigint())
            loop.run_until_complete(asyncio.gather(*all_cases, return_exceptions=True))
        except (Exception, KeyboardInterrupt) as e:
            if type(e) in ABORT_REASONS:
                # When the KeyboardInterrupt happens while asyncio.sleep it comes here
                abortall()
            else:
                getlogger().info(f"Execution stopped due to an error: {sys.exc_info()}")
        finally:
            loop.run_until_complete(_cancel_gracefully(all_cases))
            loop.close()
        loop.close()
        self._exit()


async def _cancel_gracefully(all_cases):
    for case in all_cases:
        case.cancel()
    await asyncio.gather(*all_cases, return_exceptions=True)

async def abortall_signalhandler():
    """Cancel all running tasks when SIGINT is received."""
    # Retrieve all tasks
    tasks = [t for t in asyncio.Task.all_tasks() if t is not current_task()]
    # Tasks with running subprocesses
    tasks_wait = [t for t in tasks if (hasattr(t, 'emitted_cmd'))]
    # Tasks that can be cancelled directly
    tasks_cancel = [t for t in tasks if (not hasattr(t, 'emitted_cmd') and 'runcase' in t._coro.__name__)]
    getlogger().debug2("Cancelling {0} tasks: {1}".format(len(tasks_cancel), tasks_cancel))
    getlogger().debug2("Waiting to cancel {0} tasks: {1}".format(len(tasks_wait), tasks_wait))
    # Cancel the tasks
    for task in tasks_cancel:
        print(f"Cancelando {task}")
        task.cancel()
    tasks_to_cancel = []
    for task in tasks_wait:
        task.aborting = True
        tasks_to_cancel.append(task)
    await asyncio.gather(*tasks_to_cancel, return_exceptions=False)  # Wait for them to finish
    await asyncio.gather(*tasks_cancel, return_exceptions=False)

def abortall():
    """Cancel all running tasks when SIGINT is received."""
    # Retrieve all tasks
    tasks = [t for t in asyncio.Task.all_tasks() if t is not current_task()]
    # Tasks with running subprocesses
    tasks_wait = [t for t in tasks if (hasattr(t, 'emitted_cmd'))]
    # Tasks that can be cancelled directly
    tasks_cancel = [t for t in tasks if (not hasattr(t, 'emitted_cmd') and 'runcase' in t._coro.__name__)]
    getlogger().debug2("Cancelling {0} tasks: {1}".format(len(tasks_cancel), tasks_cancel))
    getlogger().debug2("Waiting to cancel {0} tasks: {1}".format(len(tasks_wait), tasks_wait))
    # Cancel the tasks
    for task in tasks_cancel:
        task.cancel()
    tasks_to_cancel = []
    for task in tasks_wait:
        task.aborting = True
        tasks_to_cancel.append(task)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=False))  # Wait for them to finish

async def handle_sigint():
    """Handle SIGINT and cancel tasks."""
    # print(sys.exc_info())
    print("\nReceived SIGINT. Cleaning up...")
    await abortall_signalhandler()
    print("Cleanup done. Exiting.")
