import itertools
import time
import sys
import reframe.core.debug as debug

from reframe.core.exceptions import ReframeFatalError
from reframe.core.logging import getlogger
from reframe.frontend.executors import (ExecutionPolicy,
                                        RegressionTestExecutor,
                                        TestCase)
from reframe.frontend.statistics import TestStats
from reframe.settings import settings
from reframe.core.environments import EnvironmentSnapshot


class SerialExecutionPolicy(ExecutionPolicy):
    def __init__(self):
        super().__init__()
        self._test_cases = []

    def getstats(self):
        return TestStats(self._test_cases)

    def run_check(self, check, partition, environ):
        self.printer.status(
            'RUN', "%s on %s using %s" %
            (check.name, partition.fullname, environ.name)
        )
        try:
            executor = RegressionTestExecutor(check, self.strict_check)
            testcase = TestCase(executor)

            executor.setup(
                partition=partition,
                environ=environ,
                sched_account=self.sched_account,
                sched_partition=self.sched_partition,
                sched_reservation=self.sched_reservation,
                sched_nodelist=self.sched_nodelist,
                sched_exclude=self.sched_exclude_nodelist,
                sched_options=self.sched_options
            )

            executor.compile()
            executor.run()
            executor.wait()
            if not self.skip_sanity_check:
                if not executor.check_sanity():
                    testcase.fail()

            if not self.skip_performance_check:
                if not executor.check_performance():
                    testcase.fail()

            if testcase.failed():
                remove_stage_files = False
            else:
                remove_stage_files = not self.keep_stage_files

            executor.cleanup(remove_files=remove_stage_files,
                             unload_env=False)
            if not testcase.failed():
                testcase.success()

        except (KeyboardInterrupt, ReframeFatalError, AssertionError):
            testcase.fail(sys.exc_info())
            raise
        except:
            testcase.fail(sys.exc_info())
        finally:
            self._test_cases.append(testcase)
            self.printer.result(check, partition, environ,
                                not testcase.failed())
            self.environ_snapshot.load()


class RunningTestCase:
    def __init__(self, testcase, environ):
        self._testcase = testcase
        self._environ  = environ

        # Test case has finished, but has not been waited for yet
        self.zombie = False

    def __repr__(self):
        return debug.repr(self)

    @property
    def testcase(self):
        return self._testcase

    @property
    def environ(self):
        return self._environ


class WaitError(BaseException):
    """Mark wait errors during the asynchronous execution of test cases.

    It stores the `RunningTestCase` that has failed during waiting and the
    associated exception info."""

    def __init__(self, running_testcase, exc_info):
        self._running_case = running_testcase
        self._exc_info = exc_info

    @property
    def running_case(self):
        return self._running_case

    @property
    def exc_info(self):
        return self._exc_info


class AsynchronousExecutionPolicy(ExecutionPolicy):
    def __init__(self):
        super().__init__()
        # all currently running cases
        self._running_cases = []

        # counts of running cases per partition
        self._running_cases_counts = {}

        # ready cases to be executed per partition
        self._ready_cases = {}

        # Test case results
        self._test_cases = []

        # Job limit per partition
        self._max_jobs = {}

    def _compile_run_testcase(self, testcase):
        try:
            executor = testcase.executor
            executor.compile()
            executor.run()
        except (KeyboardInterrupt, ReframeFatalError, AssertionError):
            testcase.fail(sys.exc_info())
            raise
        except:
            testcase.fail(sys.exc_info())
        finally:
            if testcase.valid():
                self.printer.result(executor.check,
                                    executor.check.current_partition,
                                    executor.check.current_environ,
                                    not testcase.failed())

    def _finalize_testcase(self, ready_testcase):
        try:
            ready_testcase.environ.load()
            testcase = ready_testcase.testcase
            executor = testcase.executor
            if not self.skip_sanity_check:
                if not executor.check_sanity():
                    testcase.fail()

            if not self.skip_performance_check:
                if not executor.check_performance():
                    testcase.fail()

            if testcase.failed():
                remove_stage_files = False
            else:
                remove_stage_files = not self.keep_stage_files

            executor.cleanup(remove_files=remove_stage_files, unload_env=False)
            if not testcase.failed():
                testcase.success()

        except (KeyboardInterrupt, ReframeFatalError, AssertionError):
            testcase.fail(sys.exc_info())
            raise
        except:
            testcase.fail(sys.exc_info())
        finally:
            partname = executor.check.current_partition.fullname
            self.printer.result(executor.check,
                                executor.check.current_partition,
                                executor.check.current_environ,
                                not testcase.failed())

    def _failall(self):
        """Mark all tests as failures"""
        for rc in self._running_cases:
            rc.testcase.fail(sys.exc_info())

        for ready_list in self._ready_cases.values():
            for rc in ready_list:
                rc.testcase.fail(sys.exc_info())

    def enter_partition(self, c, p):
        self._running_cases_counts.setdefault(p.fullname, 0)
        self._ready_cases.setdefault(p.fullname, [])
        self._max_jobs.setdefault(p.fullname, p.max_jobs)

    def getstats(self):
        return TestStats(self._test_cases)

    def _print_executor_status(self, status, executor):
        checkname = executor.check.name
        partname  = executor.check.current_partition.fullname
        envname   = executor.check.current_environ.name
        msg = '%s on %s using %s' % (checkname, partname, envname)
        self.printer.status(status, msg)

    def run_check(self, check, partition, environ):
        try:
            executor = RegressionTestExecutor(check, self.strict_check)
            testcase = TestCase(executor)

            executor.setup(
                partition=partition,
                environ=environ,
                sched_account=self.sched_account,
                sched_partition=self.sched_partition,
                sched_reservation=self.sched_reservation,
                sched_nodelist=self.sched_nodelist,
                sched_exclude=self.sched_exclude_nodelist,
                sched_options=self.sched_options
            )

            ready_testcase = RunningTestCase(testcase, EnvironmentSnapshot())
            partname = partition.fullname
            if self._running_cases_counts[partname] >= partition.max_jobs:
                # Make sure that we still exceeded the job limit
                getlogger().debug('reached job limit (%s) for partition %s' %
                                  (partition.max_jobs, partname))
                self._update_running_counts()

            if self._running_cases_counts[partname] < partition.max_jobs:
                # Test's environment is already loaded; no need to be reloaded
                self._reschedule(ready_testcase, load_env=False)
            else:
                self._print_executor_status('HOLD', executor)
                self._ready_cases[partname].append(ready_testcase)

        except (KeyboardInterrupt, ReframeFatalError, AssertionError):
            if not testcase.failed():
                # test case failed during setup
                testcase.fail(sys.exc_info())
            self._failall()
            raise
        except:
            # Here we are sure that test case has failed during setup, since
            # _compile_and_run() handles already non-fatal exceptions. Though
            # we check again the testcase, just in case.
            if not testcase.failed():
                testcase.fail(sys.exc_info())
        finally:
            if testcase.valid() and testcase.failed_stage == 'setup':
                # We need to print the result here only if the setup stage has
                # finished, since otherwise _compile_and_run() prints it
                self.printer.result(executor.check, partition, environ,
                                    not testcase.failed())

            self._test_cases.append(testcase)
            self.environ_snapshot.load()

    def _update_running_counts(self):
        """Update the counts of running checks per partition."""
        getlogger().debug('updating counts for running test cases')
        freed_slots = {}
        for rc in self._running_cases:
            executor = rc.testcase.executor
            check = executor.check
            if not rc.zombie and executor.poll():
                # Tests without a job descriptor are considered finished
                rc.zombie = True
                partname = check.current_partition.fullname
                self._running_cases_counts[partname] -= 1
                freed_slots.setdefault(partname, 0)
                freed_slots[partname] += 1

        for p, ns in freed_slots.items():
            getlogger().debug('freed %s slot(s) on partition %s' % (ns, p))

    def _reschedule(self, ready_testcase, load_env=True):
        getlogger().debug('scheduling test case for running')
        testcase = ready_testcase.testcase
        executor = testcase.executor
        partname = executor.check.current_partition.fullname

        # Restore the test case's environment and run it
        if load_env:
            ready_testcase.environ.load()

        self._print_executor_status('RUN', executor)
        self._compile_run_testcase(testcase)
        if not testcase.failed():
            self._running_cases_counts[partname] += 1
            self._running_cases.append(ready_testcase)

    def _reschedule_all(self):
        self._update_running_counts()
        for partname, num_jobs in self._running_cases_counts.items():
            assert(num_jobs >= 0)
            num_empty_slots = self._max_jobs[partname] - num_jobs
            num_schedule_jobs = min([
                num_empty_slots, len(self._ready_cases[partname])
            ])

            if num_schedule_jobs:
                getlogger().debug('rescheduling %s job(s) on %s' %
                                  (num_schedule_jobs, partname))

            for i in range(num_schedule_jobs):
                ready_case = self._ready_cases[partname].pop()
                ready_case.environ.load()
                self._reschedule(ready_case)

    def _waitany(self):
        intervals = itertools.cycle(settings.job_state_poll_intervals)
        while True:
            for i, running in enumerate(self._running_cases):
                testcase = running.testcase
                executor = testcase.executor
                running_check = executor.check
                if executor.poll():
                    try:
                        executor.wait()
                        return running
                    except (KeyboardInterrupt, ReframeFatalError, AssertionError):
                        # These errors should be propagated as-is
                        testcase.fail(sys.exc_info())
                        raise
                    except:
                        testcase.fail(sys.exc_info())
                        raise WaitError(running, sys.exc_info())
                    finally:
                        # Case is no more running; update our logs
                        del self._running_cases[i]
                        if not running.zombie:
                            partname = running_check.current_partition.fullname
                            self._running_cases_counts[partname] -= 1

                            # This is just for completeness; the case is no
                            # more a zombie, since it has been waited for
                            running.zombie = False

                        if testcase.valid():
                            self.printer.result(
                                executor.check,
                                executor.check.current_partition,
                                executor.check.current_environ,
                                not testcase.failed())

            time.sleep(next(intervals))

    def exit(self):
        self.printer.separator(
            'short single line', 'waiting for spawned checks'
        )

        while len(self._running_cases):
            try:
                ready_testcase = self._waitany()
                self._finalize_testcase(ready_testcase)
                self._reschedule_all()
            except (KeyboardInterrupt, ReframeFatalError, AssertionError):
                self._failall()
                raise
            except WaitError:
                pass
            finally:
                self.environ_snapshot.load()

        self.printer.separator(
            'short single line', 'all spawned checks finished'
        )
