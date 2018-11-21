import abc
import sys

import reframe.core.debug as debug
import reframe.core.logging as logging
import reframe.core.runtime as runtime
from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import (AbortTaskError, JobNotStartedError,
                                     ReframeFatalError, TaskExit)
from reframe.frontend.printer import PrettyPrinter
from reframe.frontend.statistics import TestStats
from reframe.utility.sandbox import Sandbox

ABORT_REASONS = (KeyboardInterrupt, ReframeFatalError, AssertionError)


class RegressionTask:
    """A class representing a :class:`RegressionTest` through the regression
    pipeline."""

    def __init__(self, check, listeners=[]):
        self._check = check
        self._failed_stage = None
        self._current_stage = None
        self._exc_info = (None, None, None)
        self._environ = None
        self._listeners = list(listeners)

        # Test case has finished, but has not been waited for yet
        self.zombie = False

    @property
    def check(self):
        return self._check

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
            with logging.logging_context(self._check) as logger:
                logger.debug('entering stage: %s' % self._current_stage)
                return fn(*args, **kwargs)
        except ABORT_REASONS:
            self.fail()
            raise
        except BaseException as e:
            self.fail()
            raise TaskExit from e

    def setup(self, *args, **kwargs):
        self._safe_call(self._check.setup, *args, **kwargs)
        self._environ = EnvironmentSnapshot()

    def compile(self):
        self._safe_call(self._check.compile)

    def compile_wait(self):
        self._safe_call(self._check.compile_wait)

    def run(self):
        self._safe_call(self._check.run)
        self._notify_listeners('on_task_run')

    def wait(self):
        self._safe_call(self._check.wait)
        self.zombie = False

    def poll(self):
        finished = self._safe_call(self._check.poll)
        if finished:
            self.zombie = True
            self._notify_listeners('on_task_exit')

        return finished

    def sanity(self):
        self._safe_call(self._check.sanity)

    def performance(self):
        self._safe_call(self._check.performance)

    def cleanup(self, *args, **kwargs):
        self._safe_call(self._check.cleanup, *args, **kwargs)
        self._notify_listeners('on_task_success')

    def fail(self, exc_info=None):
        self._failed_stage = self._current_stage
        self._exc_info = exc_info or sys.exc_info()
        self._notify_listeners('on_task_failure')

    def resume(self):
        self._environ.load()

    def abort(self, cause=None):
        logging.getlogger().debug('aborting: %s' % self._check.info())
        exc = AbortTaskError()
        exc.__cause__ = cause
        try:
            # FIXME: we should perhaps extend the RegressionTest interface
            # for supporting job cancelling
            if not self.zombie and self._check.job:
                self._check.job.cancel()
        except JobNotStartedError:
            self.fail((type(exc), exc, None))
        except BaseException:
            self.fail()
        else:
            self.fail((type(exc), exc, None))


class TaskEventListener:
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
        self._sandbox = Sandbox()
        self._environ_snapshot = EnvironmentSnapshot()

    def __repr__(self):
        return debug.repr(self)

    @property
    def policy(self):
        return self._policy

    @property
    def stats(self):
        return self._stats

    def runall(self, checks):
        try:
            self._printer.separator('short double line',
                                    'Running %d check(s)' % len(checks))
            self._printer.timestamp('Started on', 'short double line')
            self._printer.info()
            self._runall(checks)
            if self._max_retries:
                self._retry_failed(checks)

        finally:
            # Print the summary line
            num_failures = self._stats.num_failures()
            num_cases    = self._stats.num_cases(run=0)
            self._printer.status(
                'FAILED' if num_failures else 'PASSED',
                'Ran %d test case(s) from %d check(s) (%d failure(s))' %
                (num_cases, len(checks), num_failures), just='center'
            )
            self._printer.timestamp('Finished on', 'short double line')
            self._environ_snapshot.load()

    def _partition_supported(self, check, partition):
        if self._policy.skip_system_check:
            return True

        return check.supports_system(partition.name)

    def _environ_supported(self, check, environ):
        ret = True
        if self._policy.only_environs:
            ret = environ.name in self._policy.only_environs

        if self._policy.skip_environ_check:
            return ret
        else:
            return ret and check.supports_environ(environ.name)

    def _retry_failed(self, checks):
        rt = runtime.runtime()
        while (self._stats.num_failures() and
               rt.current_run < self._max_retries):
            failed_checks = [
                c for c in checks if c.name in
                set([t.check.name for t in self._stats.tasks_failed()])
            ]
            rt.next_run()

            self._printer.separator(
                'short double line',
                'Retrying %d failed check(s) (retry %d/%d)' %
                (len(failed_checks), rt.current_run, self._max_retries)
            )
            self._runall(failed_checks)

    def _runall(self, checks):
        system = runtime.runtime().system
        self._policy.enter()
        for c in checks:
            self._policy.enter_check(c)
            for p in system.partitions:
                if not self._partition_supported(c, p):
                    self._printer.status('SKIP',
                                         'skipping %s' % p.fullname,
                                         just='center',
                                         level=logging.VERBOSE)
                    continue

                self._policy.enter_partition(c, p)
                for e in p.environs:
                    if not self._environ_supported(c, e):
                        self._printer.status('SKIP',
                                             'skipping %s for %s' %
                                             (e.name, p.fullname),
                                             just='center',
                                             level=logging.VERBOSE)
                        continue

                    self._sandbox.system  = p
                    self._sandbox.environ = e
                    self._sandbox.check   = c
                    self._policy.enter_environ(self._sandbox.check,
                                               self._sandbox.system,
                                               self._sandbox.environ)
                    self._environ_snapshot.load()
                    self._policy.run_check(self._sandbox.check,
                                           self._sandbox.system,
                                           self._sandbox.environ)
                    self._policy.exit_environ(self._sandbox.check,
                                              self._sandbox.system,
                                              self._sandbox.environ)

                self._policy.exit_partition(c, p)

            self._policy.exit_check(c)

        self._policy.exit()


class ExecutionPolicy:
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

    def enter_check(self, check):
        self.printer.separator(
            'short single line',
            'started processing %s (%s)' % (check.name, check.descr)
        )

    def exit_check(self, check):
        self.printer.separator(
            'short single line',
            'finished processing %s (%s)\n' % (check.name, check.descr)
        )

    def enter_partition(self, c, p):
        pass

    def exit_partition(self, c, p):
        pass

    def enter_environ(self, c, p, e):
        pass

    def exit_environ(self, c, p, e):
        pass

    @abc.abstractmethod
    def run_check(self, c, p, e):
        """Run a check with on a specific system partition with a specific environment.

        Keyword arguments:
        c -- the check to run.
        p -- the system partition to run the check on.
        e -- the environment to run the check with.
        """
        if self.strict_check:
            c.strict_check = True

        if self.force_local:
            c.local = True

    @abc.abstractmethod
    def getstats(self):
        """Return test case statistics of the run."""
