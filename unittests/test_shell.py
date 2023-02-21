# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import stat
import tempfile
import time

import reframe.core.shell as shell
import reframe.utility.osext as osext
from reframe.core.exceptions import SpawnedProcessError


@pytest.fixture
def script_file(tmp_path):
    script_file = tmp_path / 'script.sh'
    script_file.touch()
    return script_file


def test_generate(script_file):
    with shell.generate_script(script_file) as gen:
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
    with open(script_file) as fp:
        assert expected_output == fp.read()

    assert os.stat(script_file).st_mode & stat.S_IXUSR == stat.S_IXUSR


def test_generate_custom_filemode(script_file):
    with shell.generate_script(script_file, stat.S_IREAD) as gen:
        gen.write('x=1')

    assert stat.S_IMODE(os.stat(script_file).st_mode) == stat.S_IREAD


def test_generate_login(script_file):
    with shell.generate_script(script_file, login=True) as gen:
        gen.write('echo hello')

    expected_output = '''#!/bin/bash -l
echo hello
'''
    with open(script_file) as fp:
        assert expected_output == fp.read()


def test_write_types(script_file):
    class C:
        def __str__(self):
            return 'echo "C()"'

    with shell.generate_script(script_file) as gen:
        gen.write(['echo foo', 'echo hello'])
        gen.write('echo bar')
        gen.write(C())

    expected_output = '''#!/bin/bash
echo foo
echo hello
echo bar
echo "C()"
'''
    with open(script_file) as fp:
        assert expected_output == fp.read()


def test_trap_error(script_file):
    with shell.generate_script(script_file, trap_errors=True) as gen:
        gen.write('false')
        gen.write('echo hello')

    with pytest.raises(SpawnedProcessError) as cm:
        osext.run_command(str(script_file), check=True)

    exc = cm.value
    assert 'hello' not in exc.stdout
    assert 1 == exc.exitcode
    assert "-reframe: command `false' failed (exit code: 1)" in exc.stdout


def test_trap_exit(script_file):
    with shell.generate_script(script_file, trap_exit=True) as gen:
        gen.write('echo hello')

    completed = osext.run_command(str(script_file), check=True)
    assert 'hello' in completed.stdout
    assert 0 == completed.returncode
    assert '-reframe: script exiting with exit code: 0' in completed.stdout


def test_trap_signal(script_file):
    with shell.generate_script(script_file, trap_signals=True) as gen:
        gen.write('sleep 10')
        gen.write('echo hello')

    f_stdout = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    proc = osext.run_command_async(str(script_file), stdout=f_stdout,
                                   start_new_session=True)

    # Yield for some time to allow the script to start
    time.sleep(1)

    # We kill the whole spawned process group (we are launching a shell)
    os.killpg(proc.pid, 15)
    proc.wait()

    f_stdout.flush()
    f_stdout.seek(0)
    stdout = f_stdout.read()
    assert 'hello' not in stdout
    assert 143 == proc.returncode
    assert '-reframe: script caught signal: 15' in stdout
