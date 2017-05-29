import os
import re
import shutil
import stat

import unittest
import tempfile

import reframe.utility.os as os_ext

from reframe.frontend.loader import SiteConfiguration, autodetect_system
from unittests.fixtures import guess_system, system_with_scheduler

class TestFrontend(unittest.TestCase):
    def setUp(self):
        self.prefix     = tempfile.mkdtemp(dir='unittests')
        self.stagedir   = os.path.join(self.prefix, 'stage')
        self.outputdir  = os.path.join(self.prefix, 'output')
        self.logdir     = os.path.join(self.prefix, 'logs')
        self.python     = 'python3'
        self.executable = 'reframe.py'
        self.sysopt     = 'generic:login'
        self.checkfile  = 'unittests/resources/hellocheck.py'
        self.prgenv     = 'builtin-gcc'
        self.cmdstr     = '{python} {executable} {checkopt} ' \
                          '-o {outputdir} -s {stagedir} ' \
                          '--logdir {logdir} {prgenvopt} ' \
                          '--notimestamp --nocolor {action} ' \
                          '{sysopt} {local} {options}'
        self.options    = ''
        self.action     = '-r'
        self.local      = True

        # needed for enabling/disabling tests based on the current system
        self.system = autodetect_system(SiteConfiguration())


    def _invocation_cmd(self):
        return self.cmdstr.format(
            python     = self.python,
            executable = self.executable,
            checkopt   = ('-c %s' % self.checkfile) if self.checkfile else '',
            outputdir  = self.outputdir,
            stagedir   = self.stagedir,
            logdir     = self.logdir,
            prgenvopt  = ('-p %s' % self.prgenv) if self.prgenv else '',
            action     = self.action,
            local      = '--force-local' if self.local else '',
            options    = ' '.join(self.options),
            sysopt     = ('--system %s' % self.sysopt) if self.sysopt else ''
        )


    def _stage_exists(self, check_name, partitions, prgenv_name):
        for p in partitions:
            if not os.path.exists(os.path.join(
                    self.stagedir, p, check_name, prgenv_name)):
                return False

        return True


    def _perflog_exists(self, check_name, partitions):
        for p in partitions:
            logfile = os.path.join(self.logdir, p, check_name + '.log')
            if not os.path.exists(logfile):
                return False

        return True


    def test_unsupported_python(self):
        # The framework must make sure that an informative message is printed in
        # such case. If a SyntaxError happens, that's a problem
        self.python = 'python2'
        self.action = '-l'
        command = os_ext.run_command(self._invocation_cmd())
        self.assertIn('Unsupported Python version', command.stderr)


    def test_check_success(self):
        command = os_ext.run_command(self._invocation_cmd(), check=True)
        self.assertNotIn('FAILED', command.stdout)
        self.assertIn('PASSED', command.stdout)


    @unittest.skipIf(not system_with_scheduler(None),
                     'job submission not supported')
    def test_check_submit_success(self):
        # This test will run on the auto-detected system
        system = guess_system()
        partition = system_with_scheduler(None)

        self.local = False
        self.sysopt = '%s:%s' % (system.name, partition.name)

        # pick up the programming environment of the partition
        self.prgenv = partition.environs[0].name

        command = os_ext.run_command(self._invocation_cmd(), check=True)
        self.assertNotIn('FAILED', command.stdout)
        self.assertIn('PASSED', command.stdout)


    def test_check_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = [ '--tag BadSetupCheck' ]
        command = os_ext.run_command(self._invocation_cmd())
        self.assertIn('FAILED', command.stdout)
        self.assertNotEqual(command.returncode, 0)


    def test_check_sanity_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = [ '--tag SanityFailureCheck' ]
        command = os_ext.run_command(self._invocation_cmd())
        self.assertIn('FAILED', command.stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', command.stderr)
        self.assertNotEqual(command.returncode, 0)

        partitions = re.findall('>>>> Running regression on partition: (\S+)',
                                command.stdout)
        self.assertTrue(self._stage_exists('SanityFailureCheck',
                                           partitions, self.prgenv))


    def test_performance_check_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = [ '--tag PerformanceFailureCheck' ]
        command = os_ext.run_command(self._invocation_cmd())
        self.assertIn('FAILED', command.stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', command.stderr)
        self.assertNotEqual(command.returncode, 0)

        partitions = re.findall('>>>> Running regression on partition: (\S+)',
                                command.stdout)
        self.assertTrue(self._stage_exists('PerformanceFailureCheck',
                                           partitions, self.prgenv))
        self.assertTrue(self._perflog_exists('PerformanceFailureCheck',
                                             partitions))


    def test_custom_performance_check_failure(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = [ '--tag CustomPerformanceFailureCheck' ]
        command = os_ext.run_command(self._invocation_cmd())
        self.assertIn('FAILED', command.stdout)

        # This is a normal failure, it should not raise any exception
        self.assertNotIn('Traceback', command.stderr)
        self.assertNotEqual(command.returncode, 0)

        partitions = re.findall('>>>> Running regression on partition: (\S+)',
                                command.stdout)
        self.assertTrue(self._stage_exists('CustomPerformanceFailureCheck',
                                           partitions, self.prgenv))
        self.assertNotIn('Check log file:', command.stdout)


    def test_skip_system_check_option(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = [ '--skip-system-check', '--tag NoSystemCheck' ]
        command = os_ext.run_command(self._invocation_cmd(), check=True)
        self.assertIn('PASSED', command.stdout)


    def test_skip_prgenv_check_option(self):
        self.checkfile = 'unittests/resources/frontend_checks.py'
        self.options = [ '--skip-prgenv-check', '--tag NoPrgEnvCheck' ]
        command = os_ext.run_command(self._invocation_cmd(), check=True)
        self.assertIn('PASSED', command.stdout)


    def test_sanity_of_checks(self):
        # This test will effectively load all the tests in the checks path and
        # will force a syntactic and runtime check at least for the constructor
        # of the checks
        self.action = '-l'
        self.checkfile = None
        command = os_ext.run_command(self._invocation_cmd(), check=True)


    def test_sanity_of_optconfig(self):
        # Test the sanity of the command line options configuration
        self.action = '-h'
        self.checkfile = None
        command = os_ext.run_command(self._invocation_cmd(), check=True)


    def test_checkpath_recursion(self):
        self.action = '-l'
        self.checkfile = None
        command = os_ext.run_command(self._invocation_cmd(), check=True)
        num_checks_default = re.search(
            'Found (\d+) check', command.stdout, re.MULTILINE).group(1)

        self.checkfile = 'checks/'
        self.options = [ '-R' ]
        command = os_ext.run_command(self._invocation_cmd(), check=True)
        num_checks_in_checkdir = re.search(
            'Found (\d+) check', command.stdout, re.MULTILINE).group(1)
        self.assertEqual(num_checks_in_checkdir, num_checks_default)

        self.options = []
        command = os_ext.run_command(self._invocation_cmd(), check=True)
        num_checks_in_checkdir = re.search(
            'Found (\d+) check', command.stdout, re.MULTILINE).group(1)
        self.assertEqual('0', num_checks_in_checkdir)


    def tearDown(self):
        shutil.rmtree(self.prefix)
