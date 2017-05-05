import os
import shutil
import stat
import tempfile
import unittest

import reframe.utility.os as os_ext

from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import ReframeError, CommandError
from reframe.core.modules import *
from reframe.utility.functions import *

from unittests.fixtures import TEST_MODULES

class TestOSTools(unittest.TestCase):
    def test_command_success(self):
        completed = os_ext.run_command('echo foobar')
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, 'foobar\n')


    def test_command_error(self):
        self.assertRaises(CommandError, os_ext.run_command,
                          'false', 'check=True')


    def test_command_timeout(self):
        try:
            os_ext.run_command('sleep 3', timeout=2)
            self.fail('Expected timeout')
        except CommandError as e:
            self.assertEqual(e.timeout, 2)


    def test_command_async(self):
        from datetime import datetime

        t_launch = datetime.now()
        t_sleep  = t_launch
        proc = os_ext.run_command_async('sleep 1')
        t_launch = datetime.now() - t_launch

        proc.wait()
        t_sleep = datetime.now() - t_sleep

        # Now check the timings
        self.assertLess(t_launch.seconds, 1)
        self.assertGreaterEqual(t_sleep.seconds, 1)


    def test_grep(self):
        self.assertTrue(os_ext.grep_command_output(cmd='echo hello',
                                                     pattern='hello'))
        self.assertFalse(os_ext.grep_command_output(cmd='echo hello',
                                                      pattern='foo'))

    def test_copytree(self):
        dir_src = tempfile.mkdtemp()
        dir_dst = tempfile.mkdtemp()

        self.assertRaises(OSError, shutil.copytree, dir_src, dir_dst)
        try:
            os_ext.copytree(dir_src, dir_dst)
        except Exception as e:
            self.fail('custom copytree failed: %s' % e)

        shutil.rmtree(dir_src)
        shutil.rmtree(dir_dst)


    def test_inpath(self):
        self.assertTrue(os_ext.inpath('/foo/bin', '/bin:/foo/bin:/usr/bin'))
        self.assertFalse(os_ext.inpath('/foo/bin', '/bin:/usr/local/bin'))


    def test_subdirs(self):
        # Create a temporary directory structure
        # foo/
        #   bar/
        #     boo/
        #   goo/
        # loo/
        #   bar/
        prefix = tempfile.mkdtemp()
        os.makedirs(os.path.join(prefix, 'foo', 'bar'), exist_ok=True)
        os.makedirs(os.path.join(prefix, 'foo', 'bar', 'boo'), exist_ok=True)
        os.makedirs(os.path.join(prefix, 'foo', 'goo'), exist_ok=True)
        os.makedirs(os.path.join(prefix, 'loo', 'bar'), exist_ok=True)

        # Try to fool the algorithm by adding normal files
        os.mknod(os.path.join(prefix, 'foo', 'bar', 'file.txt'), stat.S_IFREG)
        os.mknod(os.path.join(prefix, 'loo', 'file.txt'), stat.S_IFREG)

        expected_subdirs = { prefix,
                             os.path.join(prefix, 'foo'),
                             os.path.join(prefix, 'foo', 'bar'),
                             os.path.join(prefix, 'foo', 'bar', 'boo'),
                             os.path.join(prefix, 'foo', 'goo'),
                             os.path.join(prefix, 'loo'),
                             os.path.join(prefix, 'loo', 'bar') }

        returned_subdirs = os_ext.subdirs(prefix)
        self.assertEqual([prefix], returned_subdirs)

        returned_subdirs = os_ext.subdirs(prefix, recurse=True)
        self.assertEqual(expected_subdirs, set(returned_subdirs))
        shutil.rmtree(prefix)


class TestUtilityFunctions(unittest.TestCase):
    def test_standard_threshold(self):
        self.assertTrue(standard_threshold(0.9, (1.0, -0.2, 0.2)))
        self.assertTrue(standard_threshold(0.9, (1.0, None, 0.2)))
        self.assertTrue(standard_threshold(0.9, (1.0, -0.2, None)))
        self.assertTrue(standard_threshold(0.9, (1.0, None, None)))

        self.assertFalse(standard_threshold(0.5, (1.0, -0.2, 0.2)))
        self.assertFalse(standard_threshold(0.5, (1.0, -0.2, None)))
        self.assertFalse(standard_threshold(1.5, (1.0, -0.2, 0.2)))
        self.assertFalse(standard_threshold(1.5, (1.0, None, 0.2)))

        self.assertRaises(ReframeError, standard_threshold, 0.9, 1.0)
        self.assertRaises(ReframeError, standard_threshold, 0.9, (1.0,))
        self.assertRaises(ReframeError, standard_threshold, 0.9, (1.0, None))


    def test_always_true(self):
        self.assertTrue(always_true(0, None))
        self.assertTrue(always_true(230, 321.))
        self.assertTrue(always_true('foo', 232, foo=12, bar='h'))
