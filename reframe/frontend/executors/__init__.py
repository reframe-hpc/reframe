# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import contextlib
import copy
import itertools
import os
import signal
import sys
import time
import weakref

import reframe.core.fields as fields
import reframe.core.logging as logging
import reframe.core.runtime as runtime
import reframe.frontend.dependencies as dependencies
import reframe.utility.jsonext as jsonext
import reframe.utility.typecheck as typ
from reframe.core.exceptions import (AbortTaskError,
                                     JobNotStartedError,
                                     FailureLimitError,
                                     ForceExitError,
                                     RunSessionTimeout,
                                     SkipTestError,
                                     StatisticsError,
                                     TaskExit)
from reframe.core.schedulers.local import LocalJobScheduler
from reframe.frontend.printer import PrettyPrinter

ABORT_REASONS = (AssertionError, FailureLimitError,
                 KeyboardInterrupt, ForceExitError, RunSessionTimeout)


class TestStats:
    '''Stores test case statistics.'''

    def __init__(self):
        # Tasks per run stored as follows: [[run0_tasks], [run1_tasks], ...]
        self._alltasks = [[]]

    def add_task(self, task):
        current_run = runtime.runtime().current_run
        if current_run == len(self._alltasks):
            self._alltasks.append([])

        self._alltasks[current_run].append(task)

    def runs(self):
        for runid, tasks in enumerate(self._alltasks):
            yield runid, tasks

    def tasks(self, run=-1):
        if run is None:
            yield from itertools.chain(*self._alltasks)
        else:
            try:
                yield from self._alltasks[run]
            except IndexError:
                raise StatisticsError(f'no such run: {run}') from None

    def failed(self, run=-1):
        return [t for t in self.tasks(run) if t.failed]

    def skipped(self, run=-1):
        return [t for t in self.tasks(run) if t.skipped]

    def aborted(self, run=-1):
        return [t for t in self.tasks(run) if t.aborted]

    def completed(self, run=-1):
        return [t for t in self.tasks(run) if t.completed]

    def num_cases(self, run=-1):
        return sum(1 for _ in self.tasks(run))

    @property
    def num_runs(self):
        return len(self._alltasks)


class TestCase:
    '''A combination of a regression check, a system partition
    and a programming environment.
    '''

    def __init__(self, check, partition, environ):
        self._check_orig = check
        self._check = check
        self._partition = partition
        self._environ = environ
        self._deps = []
        self._is_ready = False

        # Incoming dependencies
        self.in_degree = 0

        # Level in the dependency chain
        self.level = 0

    def __iter__(self):
        # Allow unpacking a test case with a single liner:
        #       c, p, e = case
        return iter([self._check, self._partition, self._environ])

    def __hash__(self):
        return (hash(self.check.unique_name) ^
                hash(self.partition.fullname) ^
                hash(self.environ.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self.check.unique_name == other.check.unique_name and
                self.environ.name == other.environ.name and
                self.partition.fullname == other.partition.fullname)

    def __repr__(self):
        c = self.check.unique_name if self.check else None
        p = self.partition.fullname  if self.partition else None
        e = self.environ.name if self.environ else None
        return f'({c!r}, {p!r}, {e!r})'

    def __str__(self):
        check, partition, environ = self
        return f'{check.name} @{partition.fullname}+{environ.name}'

    def prepare(self):
        '''Prepare test case for sending down the test pipeline'''
        if self._is_ready:
            return

        self._check = copy.deepcopy(self._check)
        self._check._case = weakref.ref(self)
        self._is_ready = True

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


@logging.time_function
def generate_testcases(checks, prepare=False):
    '''Generate concrete test cases from checks.

    If `prepare` is true then each of the cases will also be prepared for
    being sent to the test pipeline. Note that setting this to true may slow
    down the test case generation.
    '''

    cases = []
    for c in checks:
        valid_comb = runtime.valid_sysenv_comb(c.valid_systems,
                                               c.valid_prog_environs)
        for part, environs in valid_comb.items():
            for env in environs:
                case = TestCase(c, part, env)
                if prepare:
                    case.prepare()

                cases.append(case)

    return cases


def clone_testcases(cases):
    new_cases = [tc.clone() for tc in cases]
    return dependencies.toposort(dependencies.build_deps(new_cases)[0])


class RegressionTask:
    '''A class representing a :class:`RegressionTest` through the regression
    pipeline.'''

    def __init__(self, case, listeners=None, timeout=None):
        self._case = case
        self._failed_stage = None
        self._current_stage = 'startup'
        self._exc_info = (None, None, None)
        self._listeners = listeners or []
        self._skipped = False

        # Reference count for dependent tests; safe to cleanup the test only
        # if it is zero
        self.ref_count = case.num_dependents

        # Test case has finished, but has not been waited for yet
        self.zombie = False

        # Timestamps for the start and finish phases of the pipeline
        self._timestamps = {}

        self._aborted = False

        # Performance logging
        self._perflogger = logging.null_logger
        self._perflog_compat = runtime.runtime().get_option(
            'logging/0/perflog_compat'
        )

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
    def state(self):
        if self.failed:
            return 'fail'

        if self.skipped:
            return 'skip'

        states = {
            'startup': 'startup',
            'setup': 'ready_compile',
            'compile': 'compiling',
            'compile_wait': 'ready_run',
            'run': 'running',
            'run_wait': 'completing',
            'finalize': 'retired',
            'cleanup': 'completed',
        }
        return states[self._current_stage]

    @property
    def failed_stage(self):
        return self._failed_stage

    @property
    def succeeded(self):
        return (self._current_stage in {'finalize', 'cleanup'} and
                not self._failed_stage == 'cleanup')

    @property
    def completed(self):
        return self.failed or self.succeeded

    @property
    def aborted(self):
        return self._aborted

    @property
    def skipped(self):
        return self._skipped

    @property
    def result(self):
        if self.succeeded:
            return 'pass'
        elif self.failed:
            return 'fail'
        elif self.aborted:
            return 'abort'
        elif self.skipped:
            return 'skip'
        else:
            return '<unknown>'

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
                if fn.__name__ not in ('poll',
                                       'run_complete',
                                       'compile_complete'):
                    stage = self._current_stage
                    self._timestamps[f'{stage}_start'] = time.time()

            def __exit__(this, exc_type, exc_value, traceback):
                stage = self._current_stage
                self._timestamps[f'{stage}_finish'] = time.time()
                self._timestamps['pipeline_end'] = time.time()

        if fn.__name__ not in ('poll', 'run_complete', 'compile_complete'):
            self._current_stage = fn.__name__

        try:
            with logging.logging_context(self.check) as logger:
                logger.debug(f'Entering stage: {self._current_stage}')
                with update_timestamps():
                    # Pick the configuration of the current partition
                    with runtime.temp_config(self.testcase.partition.fullname):
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

    def _dry_run_call(self, fn, *args, **kwargs):
        '''Call check's fn method in dry-run mode.'''

        @contextlib.contextmanager
        def temp_dry_run(check):
            dry_run_save = check._rfm_dry_run
            try:
                check._rfm_dry_run = True
                yield check
            except ABORT_REASONS:
                raise
            except BaseException:
                pass
            finally:
                check._rfm_dry_run = dry_run_save

        with runtime.temp_config(self.testcase.partition.fullname):
            with temp_dry_run(self.check):
                return fn(*args, **kwargs)

    @logging.time_function
    def setup(self, *args, **kwargs):
        self.testcase.prepare()
        self._safe_call(self.check.setup, *args, **kwargs)
        self._notify_listeners('on_task_setup')

    @logging.time_function
    def compile(self):
        self._safe_call(self.check.compile)
        self._notify_listeners('on_task_compile')

    @logging.time_function
    def compile_wait(self):
        self._safe_call(self.check.compile_wait)

    @logging.time_function
    def run(self):
        self._safe_call(self.check.run)
        self._notify_listeners('on_task_run')

    @logging.time_function
    def run_complete(self):
        done = self._safe_call(self.check.run_complete)
        if done:
            self.zombie = True
            self._notify_listeners('on_task_exit')

        return done

    @logging.time_function
    def compile_complete(self):
        done = self._safe_call(self.check.compile_complete)
        if done:
            self._notify_listeners('on_task_compile_exit')

        return done

    @logging.time_function
    def run_wait(self):
        self._safe_call(self.check.run_wait)
        self.zombie = False

    @logging.time_function
    def sanity(self):
        self._perflogger = logging.getperflogger(self.check)
        self._safe_call(self.check.sanity)

    @logging.time_function
    def performance(self, dry_run=False):
        if dry_run:
            self._dry_run_call(self.check.performance)
        else:
            self._safe_call(self.check.performance)

    @logging.time_function
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
        self._perflogger.log_performance(logging.INFO, self,
                                         multiline=self._perflog_compat)

    @logging.time_function
    def cleanup(self, *args, **kwargs):
        self._safe_call(self.check.cleanup, *args, **kwargs)

    def fail(self, exc_info=None, callback='on_task_failure'):
        self._failed_stage = self._current_stage
        self._exc_info = exc_info or sys.exc_info()
        self._notify_listeners(callback)
        self._perflogger.log_performance(logging.INFO, self,
                                         multiline=self._perflog_compat)

    def skip(self, exc_info=None):
        self._skipped = True
        self._failed_stage = self._current_stage
        self._exc_info = exc_info or sys.exc_info()
        self._notify_listeners('on_task_skip')

    def abort(self, cause=None):
        if self.failed or self._aborted:
            return

        logging.getlogger().debug2(f'Aborting test case: {self.testcase!r}')
        exc = AbortTaskError()
        exc.__cause__ = cause
        self._aborted = True
        try:
            if not self.zombie and self.check.job:
                self.check.job.cancel()
        except JobNotStartedError:
            self.fail((type(exc), exc, None), 'on_task_abort')
        except BaseException:
            self.fail()
        else:
            self.fail((type(exc), exc, None), 'on_task_abort')

    def info(self):
        '''Return an info string about this task.'''
        name = self.check.display_name
        hashcode = self.check.hashcode
        part = self.testcase.partition.fullname
        env  = self.testcase.environ.name
        return f'{name} /{hashcode} @{part}+{env}'


class TaskEventListener(abc.ABC):
    @abc.abstractmethod
    def on_task_setup(self, task):
        '''Called whenever the setup() method of a RegressionTask is called.'''

    @abc.abstractmethod
    def on_task_run(self, task):
        '''Called whenever the run() method of a RegressionTask is called.'''

    @abc.abstractmethod
    def on_task_compile(self, task):
        '''Called whenever the compile() method of a RegressionTask is
        called.'''

    @abc.abstractmethod
    def on_task_exit(self, task):
        '''Called whenever a RegressionTask finishes.'''

    @abc.abstractmethod
    def on_task_compile_exit(self, task):
        '''Called whenever a RegressionTask compilation phase finishes.'''

    @abc.abstractmethod
    def on_task_skip(self, task):
        '''Called whenever a RegressionTask is skipped.'''

    @abc.abstractmethod
    def on_task_failure(self, task):
        '''Called when a RegressionTask has failed.'''

    @abc.abstractmethod
    def on_task_abort(self, task):
        '''Called when a RegressionTask has aborted.'''

    @abc.abstractmethod
    def on_task_success(self, task):
        '''Called when a regression test has succeeded.'''


def _handle_sigterm(signum, frame):
    raise ForceExitError('received TERM signal')


class Runner:
    '''Responsible for executing a set of regression tests based on an
    execution policy.'''

    _timeout = fields.TypedField(typ.Duration, type(None), allow_implicit=True)

    def __init__(self, policy, printer=None, max_retries=0,
                 max_failures=sys.maxsize, reruns=0, timeout=None,
                 retries_threshold=sys.maxsize):
        self._policy = policy
        self._printer = printer or PrettyPrinter()
        self._max_retries = max_retries
        self._retries_threshold = retries_threshold
        self._num_reruns = reruns
        self._timeout = timeout
        self._t_init = timeout
        self._global_stats = False
        if self._num_reruns or self._timeout:
            self._global_stats = True

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

    @logging.time_function
    def runall(self, testcases, restored_cases=None):
        num_checks = len({tc.check.unique_name for tc in testcases})
        self._printer.separator('short double line',
                                f'Running {num_checks} check(s)')
        self._printer.timestamp('Started on', 'short double line')
        self._printer.info('')
        try:
            self._t_init = time.time()
            if self._timeout:
                self._policy.set_expiry(self._t_init + self._timeout)

            self._runall(testcases)
            if (self._max_retries and
                len(self._stats.failed()) <= self._retries_threshold):
                restored_cases = restored_cases or []
                self._retry_failed(testcases + restored_cases)

            if self._timeout:
                # Repeat the session until timeout expires
                t_elapsed = time.time() - self._t_init
                while self._timeout >= t_elapsed:
                    rt = runtime.runtime()
                    rt.next_run()
                    self._runall(clone_testcases(testcases))
                else:
                    raise RunSessionTimeout('maximum session '
                                            'duration exceeded')
            else:
                # Repeat the session if requested
                for _ in range(self._num_reruns):
                    rt = runtime.runtime()
                    rt.next_run()
                    self._runall(clone_testcases(testcases))
        finally:
            # Print the summary line
            runid = None if self._global_stats else -1
            num_aborted = len(self._stats.aborted(runid))
            num_failures = len(self._stats.failed(runid))
            if num_failures > 0:
                status = 'FAILED'
            elif num_aborted > 0:
                status = 'ABORTED'
            else:
                status = 'PASSED'

            runid = None if self._global_stats else 0
            total_run = self._stats.num_cases(runid)
            total_completed = len(self._stats.completed(runid))
            total_skipped = len(self._stats.skipped(runid))
            self._printer.status(
                status,
                f'Ran {total_completed}/{total_run}'
                f' test case(s) from {num_checks} check(s) '
                f'({num_failures} failure(s), {total_skipped} skipped, '
                f'{num_aborted} aborted)',
                just='center'
            )
            self._printer.timestamp('Finished on', 'short double line')

    def _retry_failed(self, cases):
        rt = runtime.runtime()
        failures = self._stats.failed()
        while (failures and rt.current_run < self._max_retries):
            num_failed_checks = len({tc.check.unique_name for tc in failures})
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
                '%s %s (%s)' % (prefix, check.unique_name, check.descr)
            )

        self._printer.separator('short single line',
                                'start processing checks')
        self._policy.enter()
        self._printer.reset_progress(len(testcases))
        for t in testcases:
            self._policy.runcase(t)

        self._policy.exit()
        self._printer.separator('short single line',
                                'all spawned checks have finished\n')


class ExecutionPolicy(abc.ABC):
    '''Base abstract class for execution policies.

    An execution policy implements the regression check pipeline.'''

    def __init__(self):
        # Options controlling the check execution
        self.skip_sanity_check = False
        self.skip_performance_check = False
        self.keep_stage_files = False
        self.dry_run_mode = False
        self.only_environs = None
        self.printer = None

        # Local scheduler for running forced local jobs
        self.local_scheduler = LocalJobScheduler()

        # Scheduler options
        self.sched_flex_alloc_nodes = None
        self.sched_options = []

        # Task event listeners
        self.task_listeners = []
        self.stats = None

        # Expiration time
        self._t_expire = None

    def set_expiry(self, t_point):
        self._t_expire = t_point

    def timeout_expired(self):
        if self._t_expire is None:
            return False

        return time.time() >= self._t_expire

    def enter(self):
        self._num_failed_tasks = 0

    def exit(self):
        pass

    @abc.abstractmethod
    def runcase(self, case):
        '''Run a test case.'''
