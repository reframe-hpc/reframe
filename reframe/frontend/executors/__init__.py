import abc
import sys
import reframe.core.debug as debug

from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import (
    ReframeFatalError, ReframeError, SanityError
)
from reframe.core.fields import StringField, TypedField
from reframe.core.logging import logging_context
from reframe.core.pipeline import RegressionTest
from reframe.frontend.printer import PrettyPrinter
from reframe.frontend.statistics import TestStats
from reframe.utility.sandbox import Sandbox


class TestCase:
    """Test case result placeholder class."""
    STATE_SUCCESS = 0
    STATE_FAILURE = 1

    def __init__(self, executor):
        self._executor = executor
        self._result = None
        self._failed_stage = None
        self._exc_info = None

    def __repr__(self):
        return debug.repr(self)

    @property
    def executor(self):
        return self._executor

    @property
    def result(self):
        return self._result

    @property
    def failed_stage(self):
        return self._failed_stage

    @property
    def exc_info(self):
        return self._exc_info

    def valid(self):
        return self._result is not None

    def success(self):
        self._result = TestCase.STATE_SUCCESS

    def fail(self, exc_info=None):
        self._result = TestCase.STATE_FAILURE
        self._failed_stage = self.executor.current_stage
        self._exc_info = exc_info

    def failed(self):
        return self._result == TestCase.STATE_FAILURE


class RegressionTestExecutor:
    """Responsible for the execution of `RegressionTest`'s pipeline stages.

    Keeps track of the current stage and implements relaxed performance
    checking logic."""
    _check = TypedField('_check', RegressionTest)
    _current_stage = StringField('_current_stage')

    def __init__(self, check, strict_check=False):
        self._current_stage = 'init'
        self._check = check
        if strict_check:
            self._check.strict_check = True

    def __repr__(self):
        return debug.repr(self)

    @property
    def check(self):
        return self._check

    @property
    def current_stage(self):
        return self._current_stage

    def setup(self, partition, environ, **job_opts):
        self._current_stage = 'setup'
        with logging_context(check=self._check) as logger:
            logger.debug('entering setup stage')
            self._check.setup(partition, environ, **job_opts)

    def compile(self):
        self._current_stage = 'compile'
        with logging_context(check=self._check) as logger:
            logger.debug('entering compilation stage')
            self._check.compile()

    def run(self):
        self._current_stage = 'run'
        with logging_context(check=self._check) as logger:
            logger.debug('entering running stage')
            self._check.run()

    def wait(self):
        self._current_stage = 'wait'
        with logging_context(check=self._check) as logger:
            logger.debug('entering waiting stage')
            self._check.wait()

    def poll(self):
        with logging_context(check=self._check) as logger:
            logger.debug('polling check')
            ret = self._check.poll()
            return ret

    def check_sanity(self):
        # check_sanity() may be overriden by the user tests; we log this phase
        # here then
        self._current_stage = 'sanity'
        with logging_context(check=self._check) as logger:
            logger.debug('entering sanity checking stage')
            ret = self._check.check_sanity()

        return ret

    def check_performance(self):
        # check_performance() may be overriden by the user tests; we log this
        # phase here then
        self._current_stage = 'performance'
        try:
            # FIXME: the logic has become a bit ugly here in order to support
            # both sanity syntaxes. It should be simplified again as soon as
            # the old syntax is dropped.
            with logging_context(check=self._check) as logger:
                logger.debug('entering performance checking stage')
                ret = self._check.check_performance()
        except SanityError:
            # This is to handle the new sanity systax
            if self._check.strict_check:
                raise
            else:
                return True
        else:
            return True if not self._check.strict_check else ret

    def cleanup(self, remove_files=False, unload_env=True):
        self._current_stage = 'cleanup'
        with logging_context(check=self._check) as logger:
            logger.debug('entering cleanup stage')
            self._check.cleanup(remove_files, unload_env)

        self._current_stage = 'completed'


class Runner:
    """Responsible for executing a set of regression tests based on an execution
    policy."""

    def __init__(self, policy, printer=None):
        self._printer = printer or PrettyPrinter()
        self._policy = policy
        self._policy.printer = self._printer
        self._policy.runner = self
        self._sandbox = Sandbox()
        self._stats = None

    def __repr__(self):
        return debug.repr(self)

    @property
    def policy(self):
        return self._policy

    @property
    def stats(self):
        return self._stats

    def runall(self, checks, system):
        try:
            self._printer.separator('short double line',
                                    'Running %d check(s)' % len(checks))
            self._printer.timestamp('Started on', 'short double line')
            self._printer.info()
            self._runall(checks, system)
        finally:
            # Always update statistics and print the summary line
            self._stats = self._policy.getstats()
            num_failures = self._stats.num_failures()
            self._printer.status(
                'FAILED' if num_failures else 'PASSED',
                'Ran %d test case(s) from %d check(s) (%d failure(s))' %
                (self._stats.num_cases(), len(checks), num_failures),
                just='center'
            )
            self._printer.timestamp('Finished on', 'short double line')

    def _partition_supported(self, check, partition):
        if self._policy.skip_system_check:
            return True

        return check.supports_system(partition.name)

    def _environ_supported(self, check, environ):
        precond = True
        if self._policy.only_environs:
            precond = environ.name in self._policy.only_environs

        if self._policy.skip_environ_check:
            return precond
        else:
            return precond and check.supports_progenv(environ.name)

    def _runall(self, checks, system):
        self._policy.enter()
        for c in checks:
            self._policy.enter_check(c)
            for p in system.partitions:
                if not self._partition_supported(c, p):
                    self._printer.status('SKIP',
                                         'skipping %s' % p.fullname,
                                         just='center')
                    continue

                self._policy.enter_partition(c, p)
                for e in p.environs:
                    if not self._environ_supported(c, e):
                        self._printer.status('SKIP',
                                             'skipping %s for %s' %
                                             (e.name, p.fullname),
                                             just='center')
                        continue

                    self._sandbox.system  = p
                    self._sandbox.environ = e
                    self._sandbox.check   = c
                    self._policy.enter_environ(self._sandbox.check,
                                               self._sandbox.system,
                                               self._sandbox.environ)
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
        self.environ_snapshot = EnvironmentSnapshot()
        self.strict_check = False

        # Scheduler options
        self.sched_account = None
        self.sched_partition = None
        self.sched_reservation = None
        self.sched_nodelist = None
        self.sched_exclude_nodelist = None
        self.sched_options = []

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

    @abc.abstractmethod
    def getstats(self):
        """Return test case statistics of the run."""
