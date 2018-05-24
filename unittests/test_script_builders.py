import os
import stat
import tempfile
import unittest

import reframe.core.shell as builders
import reframe.utility.os_ext as os_ext


class TestShellScriptBuilder(unittest.TestCase):
    def setUp(self):
        self.script_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        os.chmod(self.script_file.name,
                 os.stat(self.script_file.name).st_mode | stat.S_IEXEC)

    def tearDown(self):
        os.remove(self.script_file.name)

    def test_bash_builder(self):
        builder = builders.BashScriptBuilder()
        builder.set_variable('var1', '13')
        builder.set_variable('var2', '2')
        builder.set_variable('foo', '33', suppress=True)
        builder.verbatim('((var3 = var1 + var2)); echo hello $var3')
        self.script_file.write(builder.finalise())
        self.script_file.close()
        self.assertTrue(
            os_ext.grep_command_output(self.script_file.name, 'hello 15'))
        self.assertFalse(
            os_ext.grep_command_output(self.script_file.name, 'foo'))
