import sys

from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import ReframeFatalError, ReframeError
from reframe.core.fields import StringField, TypedField
from reframe.core.pipeline import RegressionTest
from reframe.frontend.printer import PrettyPrinter
from reframe.frontend.statistics import TestStats
from reframe.utility.sandbox import Sandbox

class TestCase(object):
    """Test case result placeholder class."""
    STATE_SUCCESS = 0
    STATE_FAILURE = 1

    def __init__(self, executor):
        self.executor = executor
        self.result = None
        self.failed_stage = None
        self.exc_info = None

    def valid(self):
        return self.result != None

    def success(self):
        self.result = TestCase.STATE_SUCCESS

    def fail(self, exc_info = None):
        self.result = TestCase.STATE_FAILURE
        self.failed_stage  = self.executor.current_stage
        self.exc_info = exc_info

    def failed(self):
        return self.result == TestCase.STATE_FAILURE


class RegressionTestExecutor(object):
    """Responsible for the execution of `RegressionTest`'s pipeline stages.

    Keeps track of the current stage and implements relaxed performance checking
    logic."""
    check = TypedField('check', RegressionTest)
    current_stage = StringField('current_stage')

    def __init__(self, check):
        self.current_stage = 'init'
        self.check = check
        self.relax_performance_check = False

    def setup(self, system, environ, **job_opts):
        self.current_stage = 'setup'
        self.check.setup(system, environ, **job_opts)

    def compile(self):
        self.current_stage = 'compile'
        self.check.compile()

    def run(self):
        self.current_stage = 'run'
        self.check.run()

    def wait(self):
        self.current_stage = 'wait'
        self.check.wait()

    def check_sanity(self):
        # check_sanity() may be overriden by the user tests; we log this phase
        # here then
        self.current_stage = 'sanity'
        ret = self.check.check_sanity()
        self.check.logger.debug('sanity check result: %s' % ret)
        return ret

    def check_performance(self):
        # check_performance() may be overriden by the user tests; we log this
        # phase here then
        self.current_stage = 'performance'
        ret = self.check.check_performance()
        self.check.logger.debug('performance check result: %s' % ret)
        if self.check.strict_check:
            return ret

        return True if self.relax_performance_check else ret

    def cleanup(self, remove_files=False, unload_env=True):
        self.current_stage = 'cleanup'
        self.check.cleanup(remove_files, unload_env)
        self.current_stage = 'completed'


class Runner(object):
    """Responsible for executing a set of regression tests based on an execution
    policy."""
    def __init__(self, policy, printer = None):
        self.printer = PrettyPrinter() if not printer else printer
        self.policy = policy
        self.policy.printer = self.printer
        self.policy.runner = self
        self.sandbox = Sandbox()
        self.stats = None


    def runall(self, checks, system):
        try:
            self.printer.separator('short double line',
                                   'Running %d check(s)' % len(checks))
            self.printer.timestamp('Started on', 'short double line')
            self.printer.info()
            self._runall(checks, system)
        finally:
            # Always update statistics and print the summary line
            self.stats = self.policy.getstats()
            num_failures = self.stats.num_failures()
            self.printer.status(
                'FAILED' if num_failures else 'PASSED',
                'Ran %d test case(s) from %d check(s) (%d failure(s))' % \
                (self.stats.num_cases(), len(checks), num_failures),
                just='center'
            )
            self.printer.timestamp('Finished on', 'short double line')


    def _partition_supported(self, check, partition):
        if self.policy.skip_system_check:
            return True

        return check.supports_system(partition.name)


    def _environ_supported(self, check, environ):
        precond = True
        if self.policy.only_environs:
            precond = environ.name in self.policy.only_environs

        if self.policy.skip_environ_check:
            return precond
        else:
            return precond and check.supports_progenv(environ.name)


    def _runall(self, checks, system):
        self.policy.enter()
        for c in checks:
            self.policy.enter_check(c)
            for p in system.partitions:
                if not self._partition_supported(c, p):
                    self.printer.status('SKIP',
                                        'skipping %s' % p.fullname,
                                        just='center')
                    continue

                self.policy.enter_partition(c, p)
                for e in p.environs:
                    if not self._environ_supported(c, e):
                        self.printer.status('SKIP',
                                            'skipping %s for %s' % \
                                            (e.name, p.fullname),
                                            just='center')
                        continue

                    self.sandbox.system  = p
                    self.sandbox.environ = e
                    self.sandbox.check   = c
                    self.policy.enter_environ(self.sandbox.check,
                                              self.sandbox.system,
                                              self.sandbox.environ)
                    self.policy.run_check(self.sandbox.check,
                                          self.sandbox.system,
                                          self.sandbox.environ)
                    self.policy.exit_environ(self.sandbox.check,
                                             self.sandbox.system,
                                             self.sandbox.environ)

                self.policy.exit_partition(c, p)

            self.policy.exit_check(c)

        self.policy.exit()


class ExecutionPolicy(object):
    """Base abstract class for execution policies.

    An execution policy implements the regression check pipeline."""
    def __init__(self):
        # Options controlling the check execution
        self.skip_system_check = False
        self.force_local = False
        self.relax_performance_check = False
        self.skip_environ_check = False
        self.skip_sanity_check = False
        self.skip_performance_check = False
        self.keep_stage_files = False
        self.only_environs = None
        self.printer = None
        self.environ_snapshot = EnvironmentSnapshot()

        # Scheduler options
        self.sched_account = None
        self.sched_partition = None
        self.sched_reservation = None
        self.sched_nodelist = None
        self.sched_exclude_nodelist = None
        self.sched_options = []

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

    def run_check(self, c, p, e):
        raise NotImplementedError

    def getstats(self):
        """Return test case statistics of the run."""
        raise NotImplementedError
