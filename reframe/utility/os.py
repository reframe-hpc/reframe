#
# OS and shell utility functions
#

import os
import re
import shlex
import shutil
import subprocess

from reframe.core.exceptions import *

def run_command(cmd, check=False, timeout=None):
    try:
        return subprocess.run(shlex.split(cmd),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True,
                              timeout=timeout,
                              check=check)
    except subprocess.CalledProcessError as e:
        raise CommandError(command  = e.cmd,
                           stdout   = e.stdout,
                           stderr   = e.stderr,
                           exitcode = e.returncode)

    except subprocess.TimeoutExpired as e:
        raise CommandError(command  = e.cmd,
                           stdout   = e.stdout,
                           stderr   = e.stderr,
                           exitcode = None,
                           timeout  = e.timeout)


def grep_command_output(cmd, pattern, where = 'stdout'):
    completed = subprocess.run(shlex.split(cmd),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    if where == 'stdout':
        outlist = [ completed.stdout ]
    elif where == 'stderr':
        outlist = [ completed.stderr ]
    else:
        outlist = [ completed.stdout, completed.stderr ]

    for out in outlist:
        if re.search(pattern, out, re.MULTILINE):
            return True

    return False


def run_command_async(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1):
    return subprocess.Popen(args=shlex.split(cmd),
                            stdout=stdout,
                            stderr=stderr,
                            universal_newlines=True,
                            bufsize=bufsize)


def copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
             ignore_dangling_symlinks=False):
    """
    Same as shutil.copytree() but valid also if 'dst' exists, in which case it
    will first remove it and then call the standard shutil.copytree()
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)

    shutil.copytree(src, dst, symlinks, ignore, copy_function,
                    ignore_dangling_symlinks)


def inpath(entry, pathvar):
    """Check if entry is in pathvar. pathvar is a string of the form
    'entry1:entry2:entry3'
    """
    return entry in set(pathvar.split(':'))


def subdirs(dirname, recurse=False):
    """Returns a list of dirname + its subdirectories. If recurse is True,
    recursion is performed in pre-order.
    """
    dirs = []
    if os.path.isdir(dirname):
        dirs.append(dirname)
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                dirs.extend(subdirs(entry.path, recurse))

    return dirs
