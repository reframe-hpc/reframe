#
# OS and shell utility functions
#

import os
import re
import shlex
import shutil
import subprocess

from reframe.core.exceptions import *
from reframe.core.logging import getlogger


def run_command(cmd, check=False, timeout=None):
    getlogger().debug('executing OS command: ' + cmd)
    try:
        return subprocess.run(shlex.split(cmd),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True,
                              timeout=timeout,
                              check=check)
    except subprocess.CalledProcessError as e:
        raise CommandError(command=e.cmd,
                           stdout=e.stdout,
                           stderr=e.stderr,
                           exitcode=e.returncode)

    except subprocess.TimeoutExpired as e:
        raise CommandError(command=e.cmd,
                           stdout=e.stdout,
                           stderr=e.stderr,
                           exitcode=None,
                           timeout=e.timeout)


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
                      bufsize=1,
                      **popen_args):
    getlogger().debug('executing OS command asynchronously: ' + cmd)
    return subprocess.Popen(args=shlex.split(cmd),
                            stdout=stdout,
                            stderr=stderr,
                            universal_newlines=True,
                            bufsize=bufsize,
                            **popen_args)


def copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
             ignore_dangling_symlinks=False):
    """Same as shutil.copytree() but valid also if 'dst' exists.

    In this case it will first remove it and then call the standard
    shutil.copytree()."""
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
    # 2. Convert link targes to absolute paths
    # 3. Store them in a set for quick look up inside the ignore function
    link_targets = set()
    for f in file_links:
        if os.path.isabs(f):
            raise ReframeError(
                "copytree_virtual() failed: `%s': "
                "absolute paths not allowed in file_links" % f)

        target = os.path.join(src, f)
        if not os.path.exists(target):
            raise ReframeError(
                "copytree_virtual() failed: `%s' does not exist" % target)

        if os.path.commonpath([src, target]) != src:
            raise ReframeError(
                "copytree_virtual() failed: "
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
