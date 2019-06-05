import abc
import copy
import sys
import weakref

import reframe.core.debug as debug
import reframe.core.logging as logging
import reframe.core.runtime as runtime
from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import (AbortTaskError, JobNotStartedError,
                                     ReframeFatalError, TaskExit)
from reframe.frontend.printer import PrettyPrinter
from reframe.frontend.statistics import TestStats

ABORT_REASONS = (KeyboardInterrupt, ReframeFatalError, AssertionError)


class TestCase:
    """A combination of a regression check, a system partition
    and a programming environment.
    """

    def __init__(self, check, partition, environ):
        self.__check_orig = check
        self.__check = copy.deepcopy(check)
        self.__partition = copy.deepcopy(partition)
        self.__environ = copy.deepcopy(environ)
        self.__check._case = weakref.ref(self)
        self.__deps = []

    def __iter__(self):
        # Allow unpacking a test case with a single liner:
        #       c, p, e = case
        return iter([self.__check, self.__partition, self.__environ])

    def __hash__(self):
        return (hash(self.check.name) ^
                hash(self.partition.fullname) ^
                hash(self.environ.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self.check.name == other.check.name and
                self.environ.name == other.environ.name and
                self.partition.fullname == other.partition.fullname)

    def __repr__(self):
        return '(%r, %r, %r)' % (self.check.name,
                                 self.partition.fullname, self.environ.name)

    @property
    def check(self):
        return self.__check

    @property
    def partition(self):
        return self.__partition

    @property
    def environ(self):
        return self.__environ

    @property
    def deps(self):
        return self.__deps

    def clone(self):
        # Return a fresh clone, i.e., one based on the original check
        return TestCase(self.__check_orig, self.__partition, self.__environ)


def generate_testcases(checks,
                       skip_system_check=False,
                       skip_environ_check=False,
                       allowed_environs=None):
    """Generate concrete test cases from checks."""

    def supports_partition(c, p):
        return skip_system_check or c.supports_system(p.fullname)

    def supports_environ(c, e):
        return skip_environ_check or c.supports_environ(e.name)

    rt = runtime.runtime()
    cases = []
    for c in checks:
        for p in rt.system.partitions:
            if not supports_partition(c, p):
                continue

            for e in p.environs:
                if allowed_environs is None or e.name in allowed_environs:
                    if supports_environ(c, e):
                        cases.append(TestCase(c, p, e))

    return cases


class RegressionTask:
    """A class representing a :class:`RegressionTest` through the regression
    pipeline."""

    def __init__(self, case, listeners=[]):
        self._case = case
        self._failed_stage = None
        self._current_stage = None
        self._exc_info = (None, None, None)
        self._environ = None
        self._listeners = list(listeners)

        # Test case has finished, but has not been waited for yet
        self.zombie = False

    @property
    def testcase(self):
        return self._case

    @property
    def check(self):
        return self._case.check

    @property
    def exc_info(self):
        return self._exc_info

    @property
    def failed(self):
        return self._failed_stage is not None

    @property
    def failed_stage(self):
        return self._failed_stage

    def _notify_listeners(self, callback_name):
        for l in self._listeners:
            callback = getattr(l, callback_name)
            callback(self)

    def _safe_call(self, fn, *args, **kwargs):
        self._current_stage = fn.__name__
        try:
            with logging.logging_context(self.check) as logger:
                logger.debug('entering stage: %s' % self._current_stage)
                return fn(*args, **kwargs)
        except ABORT_REASONS:
            self.fail()
            raise
        except BaseException as e:
            self.fail()
            raise TaskExit from e

    def setup(self, *args, **kwargs):
        self._safe_call(self.check.setup, *args, **kwargs)
        self._environ = EnvironmentSnapshot()

    def compile(self):
        self._safe_call(self.check.compile)

    def compile_wait(self):
        self._safe_call(self.check.compile_wait)

    def run(self):
        self._safe_call(self.check.run)
        self._notify_listeners('on_task_run')

    def wait(self):
        self._safe_call(self.check.wait)
        self.zombie = False

    def poll(self):
        finished = self._safe_call(self.check.poll)
        if finished:
            self.zombie = True
            self._notify_listeners('on_task_exit')

        return finished

    def sanity(self):
        self._safe_call(self.check.sanity)

    def performance(self):
        self._safe_call(self.check.performance)

    def cleanup(self, *args, **kwargs):
        self._safe_call(self.check.cleanup, *args, **kwargs)
        self._notify_listeners('on_task_success')

    def fail(self, exc_info=None):
        self._failed_stage = self._current_stage
        self._exc_info = exc_info or sys.exc_info()
        self._notify_listeners('on_task_failure')

    def resume(self):
        self._environ.load()

    def abort(self, cause=None):
        logging.getlogger().debug('aborting: %s' % self.check.info())
        exc = AbortTaskError()
        exc.__cause__ = cause
        try:
            # FIXME: we should perhaps extend the RegressionTest interface
            # for supporting job cancelling
            if not self.zombie and self.check.job:
                self.check.job.cancel()
        except JobNotStartedError:
            self.fail((type(exc), exc, None))
        except BaseException:
            self.fail()
        else:
            self.fail((type(exc), exc, None))


class TaskEventListener(abc.ABC):
    @abc.abstractmethod
    def on_task_run(self, task):
        """Called whenever the run() method of a RegressionTask is called."""

    @abc.abstractmethod
    def on_task_exit(self, task):
        """Called whenever a RegressionTask finishes."""

    @abc.abstractmethod
    def on_task_failure(self, task):
        """Called when a regression test has failed."""

    @abc.abstractmethod
    def on_task_success(self, task):
        """Called when a regression test has succeeded."""


class Runner:
    """Responsible for executing a set of regression tests based on an
    execution policy."""

    def __init__(self, policy, printer=None, max_retries=0):
        self._policy = policy
        self._printer = printer or PrettyPrinter()
        self._max_retries = max_retries
        self._stats = TestStats()
        self._policy.stats = self._stats
        self._policy.printer = self._printer
        self._environ_snapshot = EnvironmentSnapshot()

    def __repr__(self):
        return debug.repr(self)

    @property
    def policy(self):
        return self._policy

    @property
    def stats(self):
        return self._stats

    def runall(self, testcases):
        num_checks = len({tc.check.name for tc in testcases})
        self._printer.separator('short double line',
                                'Running %d check(s)' % num_checks)
        self._printer.timestamp('Started on', 'short double line')
        self._printer.info('')
        try:
            self._runall(testcases)
            if self._max_retries:
                self._retry_failed(testcases)

        finally:
            # Print the summary line
            num_failures = len(self._stats.failures())
            self._printer.status(
                'FAILED' if num_failures else 'PASSED',
                'Ran %d test case(s) from %d check(s) (%d failure(s))' %
                (len(testcases), num_checks, num_failures), just='center'
            )
            self._printer.timestamp('Finished on', 'short double line')
            self._environ_snapshot.load()

    def _retry_failed(self, cases):
        rt = runtime.runtime()
        failures = self._stats.failures()
        while (failures and rt.current_run < self._max_retries):
            num_failed_checks = len({tc.check.name for tc in failures})
            rt.next_run()

            self._printer.separator(
                'short double line',
                'Retrying %d failed check(s) (retry %d/%d)' %
                (num_failed_checks, rt.current_run, self._max_retries)
            )
            self._runall(t.testcase.clone() for t in failures)
            failures = self._stats.failures()

    def _runall(self, testcases):
        def print_separator(check, prefix):
            self._printer.separator(
                'short single line',
                '%s %s (%s)' % (prefix, check.name, check.descr)
            )

        self._policy.enter()
        last_check = None
        for t in testcases:
            if last_check is None or last_check.name != t.check.name:
                if last_check is not None:
                    print_separator(last_check, 'finished processing')
                    self._printer.info('')

                print_separator(t.check, 'started processing')
                last_check = t.check

            self._environ_snapshot.load()
            self._policy.runcase(t)

        # Close the last visual box
        if last_check is not None:
            print_separator(last_check, 'finished processing')
            self._printer.info('')

        self._policy.exit()


class ExecutionPolicy(abc.ABC):
    """Base abstract class for execution policies.

    An execution policy implements the regression check pipeline."""

    def __init__(self):
        # Options controlling the check execution
        self.skip_system_check = False
        self.force_local = False
        self.skip_environ_check = False
        self.skip_sanity_check = False
        self.skip_performance_check = False
        self.keep_stage_files = False
        self.only_environs = None
        self.printer = None
        self.strict_check = False

        # Scheduler options
        self.sched_flex_alloc_tasks = None
        self.sched_account = None
        self.sched_partition = None
        self.sched_reservation = None
        self.sched_nodelist = None
        self.sched_exclude_nodelist = None
        self.sched_options = []

        # Task event listeners
        self.task_listeners = []

        self.stats = None

    def __repr__(self):
        return debug.repr(self)

    def enter(self):
        pass

    def exit(self):
        pass

    @abc.abstractmethod
    def runcase(self, case):
        """Run a test case."""

        if self.strict_check:
            case.check.strict_check = True

        if self.force_local:
            case.check.local = True
