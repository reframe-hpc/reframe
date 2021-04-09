# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import copy
import os
import signal
import sys
import time
import weakref

import reframe.core.logging as logging
import reframe.core.runtime as runtime
import reframe.frontend.dependencies as dependencies
import reframe.utility.jsonext as jsonext
from reframe.core.exceptions import (AbortTaskError,
                                     JobNotStartedError,
                                     FailureLimitError,
                                     ForceExitError,
                                     SkipTestError,
                                     TaskExit)
from reframe.core.schedulers.local import LocalJobScheduler
from reframe.frontend.printer import PrettyPrinter
from reframe.frontend.statistics import TestStats

ABORT_REASONS = (AssertionError, FailureLimitError,
                 KeyboardInterrupt, ForceExitError)


class TestCase:
    '''A combination of a regression check, a system partition
    and a programming environment.
    '''

    def __init__(self, check, partition, environ):
        self._check_orig = check
        self._check = copy.deepcopy(check)
        self._partition = copy.deepcopy(partition)
        self._environ = copy.deepcopy(environ)
        self._check._case = weakref.ref(self)
        self._deps = []

        # Incoming dependencies
        self.in_degree = 0

        # Level in the dependency chain
        self.level = 0

    def __iter__(self):
        # Allow unpacking a test case with a single liner:
        #       c, p, e = case
        return iter([self._check, self._partition, self._environ])

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
        c, p, e = self.check.name, self.partition.fullname, self.environ.name
        return f'({c!r}, {p!r}, {e!r})'

    @property
    def check(self):
        return self._check

    @property
    def partition(self):
        return self._partition

    @property
    def environ(self):
        return self._environ

    @property
    def deps(self):
        return self._deps

    @property
    def num_dependents(self):
        return self.in_degree

    def clone(self):
        # Return a fresh clone, i.e., one based on the original check
        return TestCase(self._check_orig, self._partition, self._environ)


def generate_testcases(checks,
                       skip_system_check=False,
                       skip_environ_check=False,
                       allowed_environs=None):
    '''Generate concrete test cases from checks.'''

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
    '''A class representing a :class:`RegressionTest` through the regression
    pipeline.'''

    def __init__(self, case, listeners=[]):
        self._case = case
        self._failed_stage = None
        self._current_stage = 'startup'
        self._exc_info = (None, None, None)
        self._listeners = list(listeners)
        self._skipped = False

        # Reference count for dependent tests; safe to cleanup the test only
        # if it is zero
        self.ref_count = case.num_dependents

        # Test case has finished, but has not been waited for yet
        self.zombie = False

        # Timestamps for the start and finish phases of the pipeline
        self._timestamps = {}

        self._aborted = False

    def duration(self, phase):
        # Treat pseudo-phases first
        if phase == 'compile_complete':
            t_start = 'compile_start'
            t_finish = 'compile_wait_finish'
        elif phase == 'run_complete':
            t_start = 'run_start'
            t_finish = 'wait_finish'
        elif phase == 'total':
            t_start = 'setup_start'
            t_finish = 'pipeline_end'
        else:
            t_start  = f'{phase}_start'
            t_finish = f'{phase}_finish'

        start = self._timestamps.get(t_start)
        if not start:
            return None

        finish = self._timestamps.get(t_finish)
        if not finish:
            finish = self._timestamps.get('pipeline_end')

        return finish - start

    def pipeline_timings(self, phases):
        def _tf(t):
            return f'{t:.3f}s' if t else 'n/a'

        msg = ''
        for phase in phases:
            if phase == 'compile_complete':
                msg += f"compile: {_tf(self.duration('compile_complete'))} "
            elif phase == 'run_complete':
                msg += f"run: {_tf(self.duration('run_complete'))} "
            else:
                msg += f"{phase}: {_tf(self.duration(phase))} "

        if msg:
            msg = msg[:-1]

        return msg

    def pipeline_timings_all(self):
        return self.pipeline_timings([
            'setup', 'compile_complete', 'run_complete',
            'sanity', 'performance', 'total'
        ])

    def pipeline_timings_basic(self):
        return self.pipeline_timings([
            'compile_complete', 'run_complete', 'total'
        ])

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
        return (self._failed_stage is not None and
                not self._aborted and not self._skipped)

    @property
    def failed_stage(self):
        return self._failed_stage

    @property
    def succeeded(self):
        return self._current_stage in {'finalize', 'cleanup'}

    @property
    def completed(self):
        return self.failed or self.succeeded

    @property
    def aborted(self):
        return self._aborted

    @property
    def skipped(self):
        return self._skipped

    def _notify_listeners(self, callback_name):
        for l in self._listeners:
            callback = getattr(l, callback_name)
            callback(self)

    def _safe_call(self, fn, *args, **kwargs):
        class update_timestamps:
            '''Context manager to set the start and finish timestamps.'''

            # We use `this` to refer to the update_timestamps object, because
            # we don't want to masquerade the self argument of our containing
            # function
            def __enter__(this):
                if fn.__name__ != 'poll':
                    stage = self._current_stage
                    self._timestamps[f'{stage}_start'] = time.time()

            def __exit__(this, exc_type, exc_value, traceback):
                stage = self._current_stage
                self._timestamps[f'{stage}_finish'] = time.time()
                self._timestamps['pipeline_end'] = time.time()

        if fn.__name__ != 'poll':
            self._current_stage = fn.__name__

        try:
            with logging.logging_context(self.check) as logger:
                logger.debug(f'Entering stage: {self._current_stage}')
                with update_timestamps():
                    return fn(*args, **kwargs)
        except SkipTestError as e:
            if not self.succeeded:
                # Only skip a test if it hasn't finished yet;
                # This practically ignores skipping during the cleanup phase
                self.skip()
                raise TaskExit from e
        except ABORT_REASONS:
            self.fail()
            raise
        except BaseException as e:
            self.fail()
            raise TaskExit from e

    def setup(self, *args, **kwargs):
        self._safe_call(self.check.setup, *args, **kwargs)
        self._notify_listeners('on_task_setup')

    def compile(self):
        self._safe_call(self.check.compile)

    def compile_wait(self):
        self._safe_call(self.check.compile_wait)

    def run(self):
        self._safe_call(self.check.run)
        self._notify_listeners('on_task_run')

    def run_complete(self):
        done = self._safe_call(self.check.run_complete)
        if done:
            self.zombie = True
            self._notify_listeners('on_task_exit')

        return done

    def run_wait(self):
        self._safe_call(self.check.run_wait)
        self.zombie = False

    def sanity(self):
        self._safe_call(self.check.sanity)

    def performance(self):
        self._safe_call(self.check.performance)

    def finalize(self):
        try:
            jsonfile = os.path.join(self.check.stagedir, '.rfm_testcase.json')
            with open(jsonfile, 'w') as fp:
                jsonext.dump(self.check, fp, indent=2)
        except OSError as e:
            logging.getlogger().warning(
                f'could not dump test case {self.testcase}: {e}'
            )

        self._current_stage = 'finalize'
        self._notify_listeners('on_task_success')

    def cleanup(self, *args, **kwargs):
        self._safe_call(self.check.cleanup, *args, **kwargs)

    def fail(self, exc_info=None):
        self._failed_stage = self._current_stage
        self._exc_info = exc_info or sys.exc_info()
        self._notify_listeners('on_task_failure')

    def skip(self, exc_info=None):
        self._skipped = True
        self._failed_stage = self._current_stage
        self._exc_info = exc_info or sys.exc_info()
        self._notify_listeners('on_task_skip')

    def abort(self, cause=None):
        if self.failed or self._aborted:
            return

        logging.getlogger().debug2('Aborting test case: {self.testcase!r}')
        exc = AbortTaskError()
        exc.__cause__ = cause
        self._aborted = True
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
    def on_task_setup(self, task):
        '''Called whenever the setup() method of a RegressionTask is called.'''

    @abc.abstractmethod
    def on_task_run(self, task):
        '''Called whenever the run() method of a RegressionTask is called.'''

    @abc.abstractmethod
    def on_task_exit(self, task):
        '''Called whenever a RegressionTask finishes.'''

    @abc.abstractmethod
    def on_task_skip(self, task):
        '''Called whenever a RegressionTask is skipped.'''

    @abc.abstractmethod
    def on_task_failure(self, task):
        '''Called when a regression test has failed.'''

    @abc.abstractmethod
    def on_task_success(self, task):
        '''Called when a regression test has succeeded.'''


def _handle_sigterm(signum, frame):
    raise ForceExitError('received TERM signal')


class Runner:
    '''Responsible for executing a set of regression tests based on an
    execution policy.'''

    def __init__(self, policy, printer=None, max_retries=0,
                 max_failures=sys.maxsize):
        self._policy = policy
        self._printer = printer or PrettyPrinter()
        self._max_retries = max_retries
        self._stats = TestStats()
        self._policy.stats = self._stats
        self._policy.printer = self._printer
        self._policy.max_failures = max_failures
        signal.signal(signal.SIGTERM, _handle_sigterm)

    @property
    def max_failures(self):
        return self._max_failures

    @property
    def max_retries(self):
        return self._max_retries

    @property
    def policy(self):
        return self._policy

    @property
    def stats(self):
        return self._stats

    def runall(self, testcases, restored_cases=None):
        num_checks = len({tc.check.name for tc in testcases})
        self._printer.separator('short double line',
                                'Running %d check(s)' % num_checks)
        self._printer.timestamp('Started on', 'short double line')
        self._printer.info('')
        try:
            self._runall(testcases)
            if self._max_retries:
                restored_cases = restored_cases or []
                self._retry_failed(testcases + restored_cases)
        finally:
            # Print the summary line
            num_failures = len(self._stats.failed())
            num_completed = len(self._stats.completed())
            num_skipped = len(self._stats.skipped())
            num_tasks = len(self._stats.tasks())
            if num_failures > 0 or num_completed + num_skipped < num_tasks:
                status = 'FAILED'
            else:
                status = 'PASSED'

            total_run = len(testcases)
            total_completed = len(self._stats.completed(0))
            total_skipped = len(self._stats.skipped(0))
            self._printer.status(
                status,
                f'Ran {total_completed}/{total_run}'
                f' test case(s) from {num_checks} check(s) '
                f'({num_failures} failure(s), {total_skipped} skipped)',
                just='center'
            )
            self._printer.timestamp('Finished on', 'short double line')

    def _retry_failed(self, cases):
        rt = runtime.runtime()
        failures = self._stats.failed()
        while (failures and rt.current_run < self._max_retries):
            num_failed_checks = len({tc.check.name for tc in failures})
            rt.next_run()

            self._printer.separator(
                'short double line',
                'Retrying %d failed check(s) (retry %d/%d)' %
                (num_failed_checks, rt.current_run, self._max_retries)
            )

            # Clone failed cases and rebuild dependencies among them
            failed_cases = [t.testcase.clone() for t in failures]
            cases_graph, _ = dependencies.build_deps(failed_cases, cases)
            failed_cases = dependencies.toposort(cases_graph, is_subgraph=True)
            self._runall(failed_cases)
            failures = self._stats.failed()

    def _runall(self, testcases):
        def print_separator(check, prefix):
            self._printer.separator(
                'short single line',
                '%s %s (%s)' % (prefix, check.name, check.descr)
            )

        self._policy.enter()
        self._printer.reset_progress(len(testcases))
        last_check = None
        for t in testcases:
            if last_check is None or last_check.name != t.check.name:
                if last_check is not None:
                    print_separator(last_check, 'finished processing')
                    self._printer.info('')

                print_separator(t.check, 'started processing')
                last_check = t.check

            self._policy.runcase(t)

        # Close the last visual box
        if last_check is not None:
            print_separator(last_check, 'finished processing')
            self._printer.info('')

        self._policy.exit()


class ExecutionPolicy(abc.ABC):
    '''Base abstract class for execution policies.

    An execution policy implements the regression check pipeline.'''

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

        # Local scheduler for running forced local jobs
        self.local_scheduler = LocalJobScheduler()

        # Scheduler options
        self.sched_flex_alloc_nodes = None
        self.sched_options = []

        # Task event listeners
        self.task_listeners = []
        self.stats = None

    def enter(self):
        self._num_failed_tasks = 0

    def exit(self):
        pass

    @abc.abstractmethod
    def runcase(self, case):
        '''Run a test case.'''

        # Pick the right subconfig for the current partition
        rt = runtime.runtime()
        _, partition, _ = case
        rt.site_config.select_subconfig(partition.fullname)
        if self.strict_check:
            case.check.strict_check = True

        if self.force_local:
            case.check.local = True
