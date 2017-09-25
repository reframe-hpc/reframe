import os
import re
import shutil
import stat
import sys
import unittest
import tempfile

import reframe.utility.os as os_ext
import reframe.core.logging as logging

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from reframe.core.environments import EnvironmentSnapshot
from reframe.frontend.loader import SiteConfiguration, autodetect_system
from reframe.settings import settings
from unittests.fixtures import guess_system, system_with_scheduler


def run_command_inline(argv, funct, *args, **kwargs):
    argv_save = sys.argv
    environ_save = EnvironmentSnapshot()
    captured_stdout = StringIO()
    captured_stderr = StringIO()
    sys.argv = argv
    try:
        with redirect_stdout(captured_stdout):
            with redirect_stderr(captured_stderr):
                exitcode = funct(*args, **kwargs)
    except SystemExit as e:
        exitcode = e.code
    finally:
        # restore environment and command-line arguments
        environ_save.load()
        sys.argv = argv_save
        return (exitcode,
                captured_stdout.getvalue(),
                captured_stderr.getvalue())


class TestFrontend(unittest.TestCase):
    def setUp(self):
        self.prefix     = tempfile.mkdtemp(dir='unittests')
        self.executable = './reframe.py'
        self.sysopt     = 'generic:login'
        self.checkfile  = 'unittests/resources/hellocheck.py'
        self.prgenv     = 'builtin-gcc'
        self.cmdstr     = '{executable} {checkopt} ' \
                          '--prefix {prefix} {prgenvopt} '    \
                          '--nocolor {action} ' \
                          '{sysopt} {local} {options}'
        self.options    = []
        self.action     = '-r'
        self.local      = True

        # needed for enabling/disabling tests based on the current system
        self.system = autodetect_system(SiteConfiguration())
        self.logfile = os.path.join(self.prefix, 'reframe.log')

        settings.logging_config = {
            'level': 'INFO',
            'handlers': {
                self.logfile : {
                    'level'     : 'DEBUG',
                    'format'    : '[%(asctime)s] %(levelname)s: '
                    '%(check_name)s: %(message)s',
                    'datefmt'   : '%FT%T',
                    'append'    : False,
                },
                '&1': {
                    'level'     : 'INFO',
                    'format'    : '%(message)s'
                },
            }
        }

    def _run_reframe(self):
        import reframe.frontend.cli as cli

        argv = self.cmdstr.format(
            executable=self.executable,
            checkopt=('-c %s' % self.checkfile) if self.checkfile else '',
            prefix=self.prefix,
            prgenvopt=('-p %s' % self.prgenv) if self.prgenv else '',
            action=self.action,
            local='--force-local' if self.local else '',
            options=' '.join(self.options),
            sysopt=('--system %s' % self.sysopt) if self.sysopt else ''
        ).split()

        return run_command_inline(argv, cli.main)

    def _stage_exists(self, check_name, partitions, prgenv_name):
        stagedir = os.path.join(self.prefix, 'stage')

        for p in partitions:
            if not os.path.exists(os.path.join(
                    stagedir, p, check_name, prgenv_name)):
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
        self.options = ['--save-log-files']
        returncode, stdout, stderr = self._run_reframe()
        self.assertNotIn('FAILED', stdout)
        self.assertIn('PASSED', stdout)
        self.assertEqual(0, returncode)
        self.assert_log_file_is_saved()

    @unittest.skipIf(not system_with_scheduler(None),
                     'job submission not supported')
    def test_check_submit_success(self):
        # This test will run on the auto-detected system
        system = guess_system()
        partition = system_with_scheduler(None)

        self.local = False
        self.sysopt = partition.fullname

        # pick up the programming environment of the partition
        self.prgenv = partition.environs[0].name

        returncode, stdout, stderr = self._run_reframe()
        self.assertNotIn('FAILED', stdout)
        self.assertIn('PASSED', stdout)
        self.assertEqual(0, returncode)

    def test_check_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = ['--tag BadSetupCheck']

        returncode, stdout, stderr = self._run_reframe()
        self.assertIn('FAILED', stdout)
        self.assertNotEqual(returncode, 0)

    def test_check_sanity_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = ['--tag SanityFailureCheck']

        returncode, stdout, stderr = self._run_reframe()
        self.assertIn('FAILED', stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', stderr)
        self.assertNotEqual(returncode, 0)
        self.assertTrue(self._stage_exists('SanityFailureCheck',
                                           ['login'], self.prgenv))

    def test_performance_check_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = ['--tag PerformanceFailureCheck']
        returncode, stdout, stderr = self._run_reframe()

        self.assertIn('FAILED', stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', stderr)
        self.assertNotEqual(0, returncode)
        self.assertTrue(self._stage_exists('PerformanceFailureCheck',
                                           ['login'], self.prgenv))
        self.assertTrue(self._perflog_exists('PerformanceFailureCheck',
                                             ['login']))

    def test_custom_performance_check_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = ['--tag CustomPerformanceFailureCheck']

        returncode, stdout, stderr = self._run_reframe()
        self.assertIn('FAILED', stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', stderr)
        self.assertNotEqual(0, returncode)

        self.assertTrue(self._stage_exists('CustomPerformanceFailureCheck',
                                           ['login'], self.prgenv))
        self.assertNotIn('Check log file:', stdout)

    def test_skip_system_check_option(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = ['--skip-system-check', '--tag NoSystemCheck']
        returncode, stdout, stderr = self._run_reframe()
        self.assertIn('PASSED', stdout)

    def test_skip_prgenv_check_option(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = ['--skip-prgenv-check', '--tag NoPrgEnvCheck']
        returncode, stdout, stderr = self._run_reframe()
        self.assertIn('PASSED', stdout)
        self.assertEqual(0, returncode)

    def test_sanity_of_checks(self):
        # This test will effectively load all the tests in the checks path and
        # will force a syntactic and runtime check at least for the constructor
        # of the checks
        self.action = '-l'
        self.options = ['--save-log-files']
        self.checkfile = None
        returncode, stdout, stderr = self._run_reframe()

        self.assertEqual(0, returncode)
        self.assert_log_file_is_saved()

    def test_unknown_system(self):
        self.action = '-l'
        self.sysopt = 'foo'
        self.checkfile = None
        returncode, stdout, stderr = self._run_reframe()
        self.assertNotIn('Traceback', stdout)
        self.assertNotIn('Traceback', stderr)
        self.assertEqual(1, returncode)

    def test_sanity_of_optconfig(self):
        # Test the sanity of the command line options configuration
        self.action = '-h'
        self.checkfile = None
        returncode, stdout, stderr = self._run_reframe()

    def test_checkpath_recursion(self):
        self.action = '-l'
        self.checkfile = None
        returncode, stdout, stderr = self._run_reframe()
        num_checks_default = re.search(
            'Found (\d+) check', stdout, re.MULTILINE).group(1)

        self.checkfile = 'checks/'
        self.options = ['-R']
        returncode, stdout, stderr = self._run_reframe()
        num_checks_in_checkdir = re.search(
            'Found (\d+) check', stdout, re.MULTILINE).group(1)
        self.assertEqual(num_checks_in_checkdir, num_checks_default)

        self.options = []
        returncode, stdout, stderr = self._run_reframe()
        num_checks_in_checkdir = re.search(
            'Found (\d+) check', stdout, re.MULTILINE).group(1)
        self.assertEqual('0', num_checks_in_checkdir)

    def test_same_output_stage_dir(self):
        output_dir = os.path.join(self.prefix, 'foo')
        self.options = ('-o %s -s %s' % (output_dir, output_dir)).split()
        returncode, stdout, stderr = self._run_reframe()
        self.assertEqual(1, returncode)

        # retry with --keep-stage-files
        self.options.append('--keep-stage-files')
        returncode, stdout, stderr = self._run_reframe()
        self.assertEqual(0, returncode)
        self.assertTrue(os.path.exists(output_dir))

    def tearDown(self):
        shutil.rmtree(self.prefix)
