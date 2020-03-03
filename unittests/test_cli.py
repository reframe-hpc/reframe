# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import itertools
import os
import pytest
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import reframe.core.config as config
import reframe.core.environments as env
import reframe.core.runtime as rt
import reframe.utility.os_ext as os_ext
import unittests.fixtures as fixtures


def run_command_inline(argv, funct, *args, **kwargs):
    # Save current execution context
    argv_save = sys.argv
    environ_save = env.snapshot()
    sys.argv = argv
    exitcode = None

    captured_stdout = StringIO()
    captured_stderr = StringIO()
    print(*sys.argv)
    with redirect_stdout(captured_stdout):
        with redirect_stderr(captured_stderr):
            try:
                with rt.temp_runtime(None):
                    exitcode = funct(*args, **kwargs)
            except SystemExit as e:
                exitcode = e.code
            finally:
                # Restore execution context
                environ_save.restore()
                sys.argv = argv_save

    return (exitcode,
            captured_stdout.getvalue(),
            captured_stderr.getvalue())


class TestFrontend(unittest.TestCase):
    @property
    def argv(self):
        ret = ['./bin/reframe', '--prefix', self.prefix, '--nocolor']
        if self.mode:
            ret += ['--mode', self.mode]

        if self.system:
            ret += ['--system', self.system]

        if self.config_file:
            ret += ['-C', self.config_file]

        ret += itertools.chain(*(['-c', c] for c in self.checkpath))
        ret += itertools.chain(*(['-p', e] for e in self.environs))

        if self.local:
            ret += ['--force-local']

        if self.action == 'run':
            ret += ['-r']
        elif self.action == 'list':
            ret += ['-l']
        elif self.action == 'list_detailed':
            ret += ['-L']
        elif self.action == 'help':
            ret += ['-h']

        if self.ignore_check_conflicts:
            ret += ['--ignore-check-conflicts']

        if self.perflogdir:
            ret += ['--perflogdir', self.perflogdir]

        ret += self.more_options
        return ret

    def setUp(self):
        self.prefix = tempfile.mkdtemp(dir='unittests')
        self.system = 'generic:login'
        self.checkpath = ['unittests/resources/checks/hellocheck.py']
        self.environs  = ['builtin-gcc']
        self.local  = True
        self.action = 'run'
        self.more_options = []
        self.mode = None
        self.config_file = 'unittests/resources/settings.py'
        self.logfile = '.rfm_unittest.log'
        self.ignore_check_conflicts = True
        self.perflogdir = '.rfm-perflogs'

    def tearDown(self):
        os_ext.rmtree(self.prefix)
        os_ext.rmtree(self.perflogdir, ignore_errors=True)
        os_ext.force_remove_file(self.logfile)

    def _run_reframe(self):
        import reframe.frontend.cli as cli
        return run_command_inline(self.argv, cli.main)

    def _stage_exists(self, check_name, partitions, environs):
        stagedir = os.path.join(self.prefix, 'stage', 'generic')
        for p in partitions:
            for e in environs:
                path = os.path.join(stagedir, p, e, check_name)
                if not os.path.exists(path):
                    return False

        return True

    def _perflog_exists(self, check_name):
        logfile = '.rfm-perflogs/generic/login/%s.log' % check_name
        return os.path.exists(logfile)

    def assert_log_file_is_saved(self):
        outputdir = os.path.join(self.prefix, 'output')
        assert os.path.exists(self.logfile)
        assert os.path.exists(
            os.path.join(outputdir, os.path.basename(self.logfile)))

    def test_default_settings(self):
        # Simply make sure that a default settings file exists
        try:
            import reframe.settings as settings
        except ImportError:
            pytest.fail('default settings file could not be found')

    def test_check_success(self):
        self.more_options = ['--save-log-files']
        returncode, stdout, _ = self._run_reframe()
        assert 'PASSED' in stdout
        assert 'FAILED' not in stdout
        assert 0 == returncode
        self.assert_log_file_is_saved()

    @fixtures.switch_to_user_runtime
    def test_check_submit_success(self):
        # This test will run on the auto-detected system
        partition = fixtures.partition_with_scheduler()
        if not partition:
            pytest.skip('job submission not supported')

        self.config_file = fixtures.USER_CONFIG_FILE
        self.local = False
        self.system = partition.fullname

        # Pick up the programming environment of the partition
        # Prepend ^ and append $ so as to much exactly the given name
        self.environs = ['^' + partition.environs[0].name + '$']

        returncode, stdout, _ = self._run_reframe()
        assert 'FAILED' not in stdout
        assert 'PASSED' in stdout

        # Assert that we have run only one test case
        assert 'Ran 1 test case(s)' in stdout
        assert 0 == returncode

    def test_check_failure(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.more_options = ['-t', 'BadSetupCheck']

        returncode, stdout, _ = self._run_reframe()
        assert 'FAILED' in stdout
        assert returncode != 0

    def test_check_setup_failure(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.more_options = ['-t', 'BadSetupCheckEarly']
        self.local = False

        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 'FAILED' in stdout
        assert returncode != 0

    def test_check_kbd_interrupt(self):
        self.checkpath = [
            'unittests/resources/checks_unlisted/kbd_interrupt.py'
        ]
        self.more_options = ['-t', 'KeyboardInterruptCheck']
        self.local = False

        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 'FAILED' in stdout
        assert returncode != 0

    def test_check_sanity_failure(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.more_options = ['-t', 'SanityFailureCheck']

        returncode, stdout, stderr = self._run_reframe()
        assert 'FAILED' in stdout

        # This is a normal failure, it should not raise any exception
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert returncode != 0
        assert self._stage_exists('SanityFailureCheck', ['login'],
                                  self.environs)

    def test_performance_check_failure(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.more_options = ['-t', 'PerformanceFailureCheck']
        returncode, stdout, stderr = self._run_reframe()

        assert 'FAILED' in stdout

        # This is a normal failure, it should not raise any exception
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 0 != returncode
        assert self._stage_exists('PerformanceFailureCheck', ['login'],
                                  self.environs)
        assert self._perflog_exists('PerformanceFailureCheck')

    def test_performance_report(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.more_options = ['-t', 'PerformanceFailureCheck',
                             '--performance-report']
        returncode, stdout, stderr = self._run_reframe()

        assert r'PERFORMANCE REPORT' in stdout
        assert r'perf: 10 Gflop/s' in stdout

    def test_skip_system_check_option(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.more_options = ['--skip-system-check', '-t', 'NoSystemCheck']
        returncode, stdout, _ = self._run_reframe()
        assert 'PASSED' in stdout

    def test_skip_prgenv_check_option(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.more_options = ['--skip-prgenv-check', '-t', 'NoPrgEnvCheck']
        returncode, stdout, _ = self._run_reframe()
        assert 'PASSED' in stdout
        assert 0 == returncode

    def test_sanity_of_checks(self):
        # This test will effectively load all the tests in the checks path and
        # will force a syntactic and runtime check at least for the constructor
        # of the checks
        self.action = 'list'
        self.more_options = ['--save-log-files']
        self.checkpath = []
        returncode, *_ = self._run_reframe()

        assert 0 == returncode
        self.assert_log_file_is_saved()

    def test_unknown_system(self):
        self.action = 'list'
        self.system = 'foo'
        self.checkpath = []
        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 1 == returncode

    def test_sanity_of_optconfig(self):
        # Test the sanity of the command line options configuration
        self.action = 'help'
        self.checkpath = []
        returncode, *_ = self._run_reframe()
        assert 0 == returncode

    def test_checkpath_colon_separated(self):
        self.action = 'list'
        self.checkpath = ['unittests/resources/checks/hellocheck_make.py:'
                          'unittests/resources/checks/hellocheck.py']
        returncode, stdout, _ = self._run_reframe()
        num_checks = re.search(
            r'Found (\d+) check', stdout, re.MULTILINE).group(1)
        assert num_checks == '2'

    def test_checkpath_symlink(self):
        self.action = 'list'
        self.checkpath = ['unittests/resources/checks']
        self.more_options = ['-R']
        returncode, stdout, _ = self._run_reframe()
        num_checks_default = re.search(
            r'Found (\d+) check', stdout, re.MULTILINE).group(1)

        with tempfile.TemporaryDirectory(dir='unittests') as tmp:
            checks_link = os.path.join(tmp, 'checks_symlink')
            os.symlink(os.path.abspath('unittests/resources/checks'),
                       os.path.abspath(checks_link))
            self.checkpath = ['unittests/resources/checks', checks_link]
            returncode, stdout, _ = self._run_reframe()
            num_checks_in_checkdir = re.search(
                r'Found (\d+) check', stdout, re.MULTILINE).group(1)

        assert num_checks_in_checkdir == num_checks_default

    def test_checkpath_recursion(self):
        self.action = 'list'
        self.checkpath = []
        returncode, stdout, _ = self._run_reframe()
        num_checks_default = re.search(
            r'Found (\d+) check', stdout, re.MULTILINE).group(1)

        self.checkpath = ['checks/']
        self.more_options = ['-R']
        returncode, stdout, _ = self._run_reframe()
        num_checks_in_checkdir = re.search(
            r'Found (\d+) check', stdout, re.MULTILINE).group(1)
        assert num_checks_in_checkdir == num_checks_default

        self.more_options = []
        returncode, stdout, stderr = self._run_reframe()
        num_checks_in_checkdir = re.search(
            r'Found (\d+) check', stdout, re.MULTILINE).group(1)
        assert '0' == num_checks_in_checkdir

    def test_same_output_stage_dir(self):
        output_dir = os.path.join(self.prefix, 'foo')
        self.more_options = ['-o', output_dir, '-s', output_dir]
        returncode, *_ = self._run_reframe()
        assert 1 == returncode

        # retry with --keep-stage-files
        self.more_options.append('--keep-stage-files')
        returncode, *_ = self._run_reframe()
        assert 0 == returncode
        assert os.path.exists(output_dir)

    def test_execution_modes(self):
        self.checkpath = []
        self.environs  = []
        self.local = False
        self.mode = 'unittest'

        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 'FAILED' not in stdout
        assert 'PASSED' in stdout
        assert 'Ran 1 test case' in stdout

    def test_no_ignore_check_conflicts(self):
        self.checkpath = ['unittests/resources/checks']
        self.more_options = ['-R']
        self.ignore_check_conflicts = False
        self.action = 'list'
        returncode, *_ = self._run_reframe()
        assert 0 != returncode

    def test_timestamp_option(self):
        from datetime import datetime

        self.checkpath = ['unittests/resources/checks']
        self.more_options = ['-R']
        self.ignore_check_conflicts = False
        self.action = 'list'
        self.more_options = ['--timestamp=xxx_%F']
        timefmt = datetime.now().strftime('xxx_%F')
        returncode, stdout, _ = self._run_reframe()
        assert 0 != returncode
        assert timefmt in stdout

    def test_list_empty_prgenvs_check_and_options(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.action = 'list'
        self.environs = []
        self.more_options = ['-n', 'NoPrgEnvCheck']
        returncode, stdout, _ = self._run_reframe()
        assert 'Found 0 check(s)' in stdout
        assert 0 == returncode

    def test_list_check_with_empty_prgenvs(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.action = 'list'
        self.environs = ['foo']
        self.more_options = ['-n', 'NoPrgEnvCheck']
        returncode, stdout, _ = self._run_reframe()
        assert 'Found 0 check(s)' in stdout
        assert 0 == returncode

    def test_list_empty_prgenvs_in_check_and_options(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.action = 'list'
        self.environs = []
        self.more_options = ['-n', 'NoPrgEnvCheck']
        returncode, stdout, _ = self._run_reframe()
        assert 'Found 0 check(s)' in stdout
        assert 0 == returncode

    def test_list_with_details(self):
        self.checkpath = ['unittests/resources/checks/frontend_checks.py']
        self.action = 'list_detailed'
        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 0 == returncode

    def test_show_config(self):
        # Just make sure that this option does not make the frontend crash
        self.more_options = ['--show-config']
        self.system = 'testsys'
        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 0 == returncode

    def test_show_env_config(self):
        # Just make sure that this option does not make the frontend crash
        self.more_options = ['--show-config-env', 'PrgEnv-gnu']
        self.system = 'testsys'
        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 0 == returncode

    def test_show_env_config_unknown_env(self):
        # Just make sure that this option does not make the frontend crash
        self.more_options = ['--show-config-env', 'foobar']
        self.system = 'testsys'
        returncode, stdout, stderr = self._run_reframe()
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 1 == returncode

    def test_verbosity(self):
        self.more_options = ['-vvvvv']
        self.system = 'testsys'
        self.action = 'list'
        returncode, stdout, stderr = self._run_reframe()
        assert '' != stdout
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert 0 == returncode

    @fixtures.switch_to_user_runtime
    def test_unload_module(self):
        # This test is mostly for ensuring coverage. `_run_reframe()` restores
        # the current environment, so it is not easy to verify that the modules
        # are indeed unloaded. However, this functionality is tested elsewhere
        # more exhaustively.

        ms = rt.runtime().modules_system
        if ms.name == 'nomod':
            pytest.skip('no modules system found')

        with rt.module_use('unittests/modules'):
            ms.load_module('testmod_foo')
            self.more_options = ['-u testmod_foo']
            self.action = 'list'
            returncode, stdout, stderr = self._run_reframe()
            ms.unload_module('testmod_foo')

        assert stdout != ''
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        assert returncode == 0
