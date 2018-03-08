import abc
import sys

import reframe.core.debug as debug
import reframe.core.logging as logging
from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import (AbortTaskError, JobNotStartedError,
                                     ReframeFatalError, TaskExit)
from reframe.frontend.printer import PrettyPrinter
from reframe.utility.sandbox import Sandbox

ABORT_REASONS = (KeyboardInterrupt, ReframeFatalError, AssertionError)


class RegressionTask:
    """A class representing a :class:`RegressionTest` through the regression
    pipeline."""

    def __init__(self, check, retry_num, listeners=[]):
        self._check = check
        self._failed_stage = None
        self._current_stage = None
        self._exc_info = (None, None, None)
        self._environ = None
        self._listeners = list(listeners)
        # TODO: if we had a runner accociated, then we could do
        # self._retry_num = self.runner.retry_num
        # Would that make sense? The runner could be passed as constructor
        # argument instead of retry_num.
        self._retry_num = retry_num

        # Test case has finished, but has not been waited for yet
        self.zombie = False

    @property
    def retry_num(self):
        return self._retry_num

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
            if not self.zombie:
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
    """Responsible for executing a set of regression tests based on an execution
    policy."""

    def __init__(self, policy, printer=None, max_retries=0):
        self._policy = policy
        self._printer = printer or PrettyPrinter()
        self._max_retries = max_retries
        self._policy.printer = self._printer
        self._policy.runner = self
        self._sandbox = Sandbox()
        self._stats = None
        self._environ_snapshot = EnvironmentSnapshot()
        self._retry_num = 0

    def __repr__(self):
        return debug.repr(self)

    @property
    def max_retries(self):
        return self._max_retries

    @property
    def retry_num(self):
        return self._retry_num

    @property
    def policy(self):
        return self._policy

    @property
    # This property function should in general be preferred over self._stats,
    # because it makes sure that the stats are updated.
    def stats(self):
        # Always update statistics
        self._stats = self._policy.getstats()
        return self._stats

    def runall(self, checks, system):
        try:
            self._printer.separator('short double line',
                                    'Running %d check(s)' % len(checks))
            self._printer.timestamp('Started on', 'short double line')
            self._printer.info()
            self._runall(checks, system)
            if self._max_retries > 0:
                self._retry(checks, system)

        finally:
            # Print the summary line
            num_failures_last_retry = self.stats.num_failures(retry_num=-1)
            self._printer.status(
                'FAILED' if num_failures_last_retry else 'PASSED',
                'Ran %d test case(s) from %d check(s) (%d failure(s) after '
                '%d retries; %d test cases passed in retries)' %
                (self.stats.num_cases(), len(checks), num_failures_last_retry,
                 self.stats.last_retry(),
                 self.stats.num_success_all_retries()),
                just='center'
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

    def _retry(self, checks, system):
        while (self.stats.num_failures() and self._retry_num <
                                             self._max_retries):
            self._retry_num += 1
            check_names_failed = self.stats.check_names_failed(retry_num=-1)
            checks_failed = []
            for check in checks:
                if check.name in check_names_failed:
                    checks_failed.append(check)

            self._runall(checks_failed, system)

    def _runall(self, checks, system):
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
        self.sched_account = None
        self.sched_partition = None
        self.sched_reservation = None
        self.sched_nodelist = None
        self.sched_exclude_nodelist = None
        self.sched_options = []

        # Task event listeners
        self.task_listeners = []

        # Associated runner
        self.runner = None

    def __repr__(self):
        return debug.repr(self)

    def enter(self):
        pass

    def exit(self):
        pass

    def enter_check(self, check):
        # TODO: Better PipelineError?
        # The following should never happen
        if self.runner is None:
            raise ReframeFatalError()

        msg = 'started processing'
        if self.runner.retry_num > 0:
            msg = 'retrying (%d/%d)' % (self.runner.retry_num,
                                        self.runner.max_retries)

        self.printer.separator(
            'short single line',
            '%s %s (%s)' % (msg, check.name, check.descr)
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

    @abc.abstractmethod
    def getstats(self):
        """Return test case statistics of the run."""
