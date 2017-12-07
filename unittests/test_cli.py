import itertools
import os
import re
import shutil
import stat
import sys
import unittest
import tempfile

import reframe.utility.os as os_ext
import reframe.core.logging as logging
import unittests.fixtures as fixtures

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from reframe.core.environments import EnvironmentSnapshot
from reframe.core.modules import init_modules_system
from reframe.frontend.loader import SiteConfiguration, autodetect_system
from reframe.settings import settings


def run_command_inline(argv, funct, *args, **kwargs):
    argv_save = sys.argv
    environ_save = EnvironmentSnapshot()
    captured_stdout = StringIO()
    captured_stderr = StringIO()
    sys.argv = argv
    exitcode = None
    with redirect_stdout(captured_stdout):
        with redirect_stderr(captured_stderr):
            try:
                exitcode = funct(*args, **kwargs)
            except SystemExit as e:
                exitcode = e.code
            finally:
                # restore environment, command-line arguments, and the native
                # modules system
                environ_save.load()
                sys.argv = argv_save
                fixtures.init_native_modules_system()

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

        ret += itertools.chain(*(['-c', c] for c in self.checkpath))
        ret += itertools.chain(*(['-p', e] for e in self.environs))

        if self.local:
            ret += ['--force-local']

        if self.action == 'run':
            ret += ['-r']
        elif self.action == 'list':
            ret += ['-l']
        elif self.action == 'help':
            ret += ['-h']

        ret += self.more_options
        return ret

    def setUp(self):
        self.prefix = tempfile.mkdtemp(dir='unittests')
        self.system = 'generic:login'
        self.checkpath = ['unittests/resources/hellocheck.py']
        self.environs  = ['builtin-gcc']
        self.local  = True
        self.action = 'run'
        self.more_options = []
        self.mode = None

        # Monkey patch logging configuration
        self.logfile = os.path.join(self.prefix, 'reframe.log')
        settings._logging_config = {
            'level': 'DEBUG',
            'handlers': {
                self.logfile: {
                    'level': 'DEBUG',
                    'format': '[%(asctime)s] %(levelname)s: '
                    '%(check_name)s: %(message)s',
                    'datefmt': '%FT%T',
                    'append': False,
                },
                '&1': {
                    'level': 'INFO',
                    'format': '%(message)s'
                },
            }
        }

        # Monkey patch site configuration setting a mode
        settings._site_configuration['modes'] = {
            '*': {
                'unittest': [
                    '-c', 'unittests/resources/hellocheck.py',
                    '-p', 'builtin-gcc',
                    '--force-local'
                ]
            }
        }

    def _run_reframe(self):
        import reframe.frontend.cli as cli
        return run_command_inline(self.argv, cli.main)

    def _stage_exists(self, check_name, partitions, environs):
        stagedir = os.path.join(self.prefix, 'stage')
        for p in partitions:
            for e in environs:
                path = os.path.join(stagedir, p, check_name, e)
                if not os.path.exists(path):
                    return False

        return True

    def _perflog_exists(self, check_name, partitions):
        logdir = os.path.join(self.prefix, 'logs')
        for p in partitions:
            logfile = os.path.join(logdir, p, check_name + '.log')
            if not os.path.exists(logfile):
                return False

        return True

    def assert_log_file_is_saved(self):
        outputdir = os.path.join(self.prefix, 'output')
        self.assertTrue(os.path.exists(self.logfile))
        self.assertTrue(os.path.exists(
            os.path.join(outputdir, os.path.basename(self.logfile))))

    def test_check_success(self):
        self.more_options = ['--save-log-files']
        returncode, stdout, _ = self._run_reframe()
        self.assertNotIn('FAILED', stdout)
        self.assertIn('PASSED', stdout)
        self.assertEqual(0, returncode)
        self.assert_log_file_is_saved()

    @unittest.skipIf(not fixtures.partition_with_scheduler(None),
                     'job submission not supported')
    def test_check_submit_success(self):
        # This test will run on the auto-detected system
        system = fixtures.HOST
        partition = fixtures.partition_with_scheduler(None)
        init_modules_system(system.modules_system)

        self.local = False
        self.system = partition.fullname

        # pick up the programming environment of the partition
        self.environs = [partition.environs[0].name]

        returncode, stdout, _ = self._run_reframe()
        self.assertNotIn('FAILED', stdout)
        self.assertIn('PASSED', stdout)
        self.assertEqual(0, returncode)

    def test_check_failure(self):
        self.checkpath = ['unittests/resources/frontend_checks.py']
        self.more_options = ['-t', 'BadSetupCheck']

        returncode, stdout, _ = self._run_reframe()
        self.assertIn('FAILED', stdout)
        self.assertNotEqual(returncode, 0)

    def test_check_setup_failure(self):
        self.checkpath = ['unittests/resources/frontend_checks.py']
        self.more_options = ['-t', 'BadSetupCheckEarlyNonLocal']
        self.local = False

        returncode, stdout, stderr = self._run_reframe()
        self.assertNotIn('Traceback', stderr)
        self.assertIn('FAILED', stdout)
        self.assertNotEqual(returncode, 0)

    def test_check_sanity_failure(self):
        self.checkpath = ['unittests/resources/frontend_checks.py']
        self.more_options = ['-t', 'SanityFailureCheck']

        returncode, stdout, stderr = self._run_reframe()
        self.assertIn('FAILED', stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', stderr)
        self.assertNotEqual(returncode, 0)
        self.assertTrue(self._stage_exists('SanityFailureCheck',
                                           ['login'], self.environs))

    def test_performance_check_failure(self):
        self.checkpath = ['unittests/resources/frontend_checks.py']
        self.more_options = ['-t', 'PerformanceFailureCheck']
        returncode, stdout, stderr = self._run_reframe()

        self.assertIn('FAILED', stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', stderr)
        self.assertNotEqual(0, returncode)
        self.assertTrue(self._stage_exists('PerformanceFailureCheck',
                                           ['login'], self.environs))
        self.assertTrue(self._perflog_exists('PerformanceFailureCheck',
                                             ['login']))

    def test_skip_system_check_option(self):
        self.checkpath = ['unittests/resources/frontend_checks.py']
        self.more_options = ['--skip-system-check', '-t', 'NoSystemCheck']
        returncode, stdout, _ = self._run_reframe()
        self.assertIn('PASSED', stdout)

    def test_skip_prgenv_check_option(self):
        self.checkpath = ['unittests/resources/frontend_checks.py']
        self.more_options = ['--skip-prgenv-check', '-t', 'NoPrgEnvCheck']
        returncode, stdout, _ = self._run_reframe()
        self.assertIn('PASSED', stdout)
        self.assertEqual(0, returncode)

    def test_sanity_of_checks(self):
        # This test will effectively load all the tests in the checks path and
        # will force a syntactic and runtime check at least for the constructor
        # of the checks
        self.action = 'list'
        self.more_options = ['--save-log-files']
        self.checkpath = []
        returncode, *_ = self._run_reframe()

        self.assertEqual(0, returncode)
        self.assert_log_file_is_saved()

    def test_unknown_system(self):
        self.action = 'list'
        self.system = 'foo'
        self.checkpath = []
        returncode, stdout, stderr = self._run_reframe()
        self.assertNotIn('Traceback', stdout)
        self.assertNotIn('Traceback', stderr)
        self.assertEqual(1, returncode)

    def test_sanity_of_optconfig(self):
        # Test the sanity of the command line options configuration
        self.action = 'help'
        self.checkpath = []
        returncode, *_ = self._run_reframe()
        self.assertEqual(0, returncode)

    def test_checkpath_recursion(self):
        self.action = 'list'
        self.checkpath = []
        returncode, stdout, _ = self._run_reframe()
        num_checks_default = re.search(
            'Found (\d+) check', stdout, re.MULTILINE).group(1)

        self.checkpath = ['checks/']
        self.more_options = ['-R']
        returncode, stdout, _ = self._run_reframe()
        num_checks_in_checkdir = re.search(
            'Found (\d+) check', stdout, re.MULTILINE).group(1)
        self.assertEqual(num_checks_in_checkdir, num_checks_default)

        self.more_options = []
        returncode, stdout, stderr = self._run_reframe()
        num_checks_in_checkdir = re.search(
            'Found (\d+) check', stdout, re.MULTILINE).group(1)
        self.assertEqual('0', num_checks_in_checkdir)

    def test_same_output_stage_dir(self):
        output_dir = os.path.join(self.prefix, 'foo')
        self.more_options = ['-o', output_dir, '-s', output_dir]
        returncode, *_ = self._run_reframe()
        self.assertEqual(1, returncode)

        # retry with --keep-stage-files
        self.more_options.append('--keep-stage-files')
        returncode, *_ = self._run_reframe()
        self.assertEqual(0, returncode)
        self.assertTrue(os.path.exists(output_dir))

    def test_execution_modes(self):
        self.checkpath = []
        self.environs  = []
        self.local = False
        self.mode = 'unittest'

        returncode, stdout, stderr = self._run_reframe()
        self.assertNotIn('Traceback', stderr)
        self.assertNotIn('FAILED', stdout)
        self.assertIn('PASSED', stdout)
        self.assertIn('Ran 1 test case', stdout)

    def tearDown(self):
        shutil.rmtree(self.prefix)
