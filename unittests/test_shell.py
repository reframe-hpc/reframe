import os
import stat
import tempfile
import time
import unittest

import reframe.core.shell as shell
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import SpawnedProcessError


class TestShellScriptGenerator(unittest.TestCase):
    def setUp(self):
        self.script_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.script_file.close()
        os.chmod(self.script_file.name,
                 os.stat(self.script_file.name).st_mode | stat.S_IEXEC)

    def tearDown(self):
        os.remove(self.script_file.name)

    def test_generate(self):
        with shell.generate_script(self.script_file.name) as gen:
            gen.write_prolog('# this is a test script')
            gen.write_prolog('# another comment')
            gen.write('v1=10')
            gen.write_prolog('export v2=20')
            gen.write_body('((v3 = v1 + v2))')
            gen.write_body('echo hello $v3')
            gen.write('unset v2')
            gen.write_epilog('echo foo')
            gen.write_epilog('unset v1')

        expected_output = '''#!/bin/bash
# this is a test script
# another comment
export v2=20
v1=10
((v3 = v1 + v2))
echo hello $v3
unset v2
echo foo
unset v1
'''
        with open(self.script_file.name) as fp:
            self.assertEqual(expected_output, fp.read())

    def test_generate_login(self):
        with shell.generate_script(self.script_file.name, login=True) as gen:
            gen.write('echo hello')

        expected_output = '''#!/bin/bash -l
echo hello
'''
        with open(self.script_file.name) as fp:
            self.assertEqual(expected_output, fp.read())

    def test_write_types(self):
        class C:
            def __str__(self):
                return 'echo "C()"'

        with shell.generate_script(self.script_file.name) as gen:
            gen.write(['echo foo', 'echo hello'])
            gen.write('echo bar')
            gen.write(C())

        expected_output = '''#!/bin/bash
echo foo
echo hello
echo bar
echo "C()"
'''
        with open(self.script_file.name) as fp:
            self.assertEqual(expected_output, fp.read())

    def test_trap_error(self):
        with shell.generate_script(self.script_file.name,
                                   trap_errors=True) as gen:
            gen.write('false')
            gen.write('echo hello')

        with self.assertRaises(SpawnedProcessError) as cm:
            os_ext.run_command(self.script_file.name, check=True)

        exc = cm.exception
        self.assertNotIn('hello', exc.stdout)
        self.assertEqual(1, exc.exitcode)
        self.assertIn("-reframe: command `false' failed (exit code: 1)",
                      exc.stdout)

    def test_trap_exit(self):
        with shell.generate_script(self.script_file.name,
                                   trap_exit=True) as gen:
            gen.write('echo hello')

        completed = os_ext.run_command(self.script_file.name, check=True)
        self.assertIn('hello', completed.stdout)
        self.assertEqual(0, completed.returncode)
        self.assertIn("-reframe: script exiting with exit code: 0",
                      completed.stdout)

    def test_trap_signal(self):
        with shell.generate_script(self.script_file.name,
                                   trap_signals=True) as gen:
            gen.write('sleep 10')
            gen.write('echo hello')

        f_stdout = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        proc = os_ext.run_command_async(self.script_file.name,
                                        stdout=f_stdout,
                                        start_new_session=True)

        # Yield for some time to allow the script to start
        time.sleep(1)

        # We kill the whole spawned process group (we are launching a shell)
        os.killpg(proc.pid, 15)
        proc.wait()

        f_stdout.flush()
        f_stdout.seek(0)
        stdout = f_stdout.read()
        self.assertNotIn('hello', stdout)
        self.assertEqual(143, proc.returncode)
        self.assertIn("-reframe: script caught signal: 15", stdout)
