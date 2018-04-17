#
# OS and shell utility functions
#

import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from urllib.parse import urlparse

from reframe.core.exceptions import (ReframeError, SpawnedProcessError,
                                     SpawnedProcessTimeout)
from reframe.core.logging import getlogger


def run_command(cmd, check=False, timeout=None, shell=False):
    try:
        proc = run_command_async(cmd, shell=shell, start_new_session=True)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        os.killpg(proc.pid, signal.SIGKILL)
        raise SpawnedProcessTimeout(e.cmd,
                                    proc.stdout.read(),
                                    proc.stderr.read(), timeout) from None

    completed = subprocess.CompletedProcess(args=shlex.split(cmd),
                                            returncode=proc.returncode,
                                            stdout=proc.stdout.read(),
                                            stderr=proc.stderr.read())

    if check and proc.returncode != 0:
        raise SpawnedProcessError(completed.args,
                                  completed.stdout, completed.stderr,
                                  completed.returncode)

    return completed


def grep_command_output(cmd, pattern, where='stdout'):
    completed = subprocess.run(shlex.split(cmd),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    if where == 'stdout':
        outlist = [completed.stdout]
    elif where == 'stderr':
        outlist = [completed.stderr]
    else:
        outlist = [completed.stdout, completed.stderr]

    for out in outlist:
        if re.search(pattern, out, re.MULTILINE):
            return True

    return False


def run_command_async(cmd,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,
                      shell=False,
                      **popen_args):
    getlogger().debug('executing OS command: ' + cmd)
    if not shell:
        cmd = shlex.split(cmd)

    return subprocess.Popen(args=cmd,
                            stdout=stdout,
                            stderr=stderr,
                            universal_newlines=True,
                            shell=shell,
                            **popen_args)


def copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
             ignore_dangling_symlinks=False):
    """Same as shutil.copytree() but valid also if 'dst' exists.

    In this case it will first remove it and then call the standard
    shutil.copytree()."""
    if src == os.path.commonpath([src, dst]):
        raise ValueError("cannot copy recursively the parent directory "
                         "`%s' into one of its descendants `%s'" % (src, dst))

    if os.path.exists(dst):
        shutil.rmtree(dst)

    shutil.copytree(src, dst, symlinks, ignore, copy_function,
                    ignore_dangling_symlinks)


def copytree_virtual(src, dst, file_links=[],
                     symlinks=False, copy_function=shutil.copy2,
                     ignore_dangling_symlinks=False):
    """Copy `dst` to `src`, but create symlinks for the files in `file_links`.

    If `file_links` is empty, this is equivalent to `copytree()`.  The rest of
    the arguments are passed as-is to `copytree()`.  Paths in `file_links` must
    be relative to `src`. If you try to pass `.` in `file_links`, `OSError`
    will be raised."""

    if not hasattr(file_links, '__iter__'):
        raise TypeError('expecting an iterable as file_links')

    # Work with absolute paths
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)

    # 1. Check that the link targets are valid
    # 2. Convert link targets to absolute paths
    # 3. Store them in a set for quick look up inside the ignore function
    link_targets = set()
    for f in file_links:
        if os.path.isabs(f):
            raise ValueError("copytree_virtual() failed: `%s': "
                             "absolute paths not allowed in file_links" % f)

        target = os.path.join(src, f)
        if not os.path.exists(target):
            raise ValueError("copytree_virtual() failed: `%s' "
                             "does not exist" % target)

        if os.path.commonpath([src, target]) != src:
            raise ValueError("copytree_virtual() failed: "
                             "`%s' not under `%s'" % (target, src))

        link_targets.add(os.path.abspath(target))

    if not file_links:
        ignore = None
    else:
        def ignore(dir, contents):
            return [c for c in contents
                    if os.path.join(dir, c) in link_targets]

    # Copy to dst ignoring the file_links
    copytree(src, dst, symlinks, ignore,
             copy_function, ignore_dangling_symlinks)

    # Now create the symlinks
    for f in link_targets:
        link_name = f.replace(src, dst)
        os.symlink(f, link_name)


def inpath(entry, pathvar):
    """Check if entry is in pathvar. pathvar is a string of the form
    `entry1:entry2:entry3`."""
    return entry in set(pathvar.split(':'))


def subdirs(dirname, recurse=False):
    """Returns a list of dirname + its subdirectories. If recurse is True,
    recursion is performed in pre-order."""
    dirs = []
    if os.path.isdir(dirname):
        dirs.append(dirname)
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                dirs.extend(subdirs(entry.path, recurse))

    return dirs


def follow_link(path):
    """Return the final target of a symlink chain"""
    while os.path.islink(path):
        path = os.readlink(path)

    return path


def samefile(path1, path2):
    """Check if paths refer to the same file.

    If paths exist, this is equivalent to `os.path.samefile()`. If only one of
    the paths exists, it will be followed if it is a symbolic link and its
    final target will be compared to the other path. If both paths do not
    exist, a simple string comparison will be performed (after they have been
    normalized)."""

    # normalise the paths first
    path1 = os.path.normpath(path1)
    path2 = os.path.normpath(path2)
    if os.path.exists(path1) and os.path.exists(path2):
        return os.path.samefile(path1, path2)

    return follow_link(path1) == follow_link(path2)


def mkstemp_path(*args, **kwargs):
    fd, path = tempfile.mkstemp(*args, **kwargs)
    os.close(fd)
    return path


class change_dir:
    """Context manager which changes the current working directory to the
       provided one."""

    def __init__(self, dir_name):
        self._wd_save = os.getcwd()
        self._dir_name = dir_name

    def __enter__(self):
        os.chdir(self._dir_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._wd_save)


def is_url(s):
    """Check if string is an URL."""
    parsed = urlparse(s)
    return parsed.scheme != '' and parsed.netloc != ''


def git_clone(url, targetdir=None):
    """Clone git repository from an URL."""
    if not git_repo_exists(url):
        raise ReframeError('git repository does not exist')

    targetdir = targetdir or ''
    run_command('git clone %s %s' % (url, targetdir), check=True)


def git_repo_exists(url, timeout=5):
    """Check if URL refers to git valid repository."""
    try:
        os.environ['GIT_TERMINAL_PROMPT'] = '0'
        run_command('git ls-remote -h %s' % url, check=True,
                    timeout=timeout)
    except (SpawnedProcessTimeout, SpawnedProcessError):
        return False
    else:
        return True
