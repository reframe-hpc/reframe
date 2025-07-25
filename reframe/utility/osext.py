# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# OS and shell utility functions
#

import collections.abc
import errno
import getpass
import grp
import os
import re
import semver
import shlex
import shutil
import signal
import sys
import subprocess
import tempfile
from urllib.parse import urlparse

import reframe
import reframe.utility as util
from reframe.core.exceptions import (ReframeError, SpawnedProcessError,
                                     SpawnedProcessTimeout)
from . import OrderedSet


class UnstartedProcError(ReframeError):
    '''Raised when a process operation is attempted on a
    not yet started process future'''


class _ProcFuture:
    '''A future encapsulating a command to be executed asynchronously.

    Users may not create a :class:`_ProcFuture` directly, but should use
    :func:`run_command_async2` instead.

    :meta public:

    .. versionadded:: 4.4
    '''

    def __init__(self, check=False, *args, **kwargs):
        self._proc = None
        self._exitcode = None
        self._signal = None
        self._check = check
        self._cmd_args = (args, kwargs)
        self._next = []
        self._done_callbacks = []
        self._completed = False
        self._cancelled = False
        self._session = False

    def _check_started(self):
        if not self.started():
            raise UnstartedProcError

    def start(self):
        '''Start the future, i.e. spawn the encapsulated command.'''

        args, kwargs = self._cmd_args
        self._proc = run_command_async(*args, **kwargs)

        try:
            if os.getsid(self._proc.pid) == self._proc.pid:
                self._session = True
        except ProcessLookupError:
            # Process has already finished
            self._wait()

        return self

    @property
    def pid(self):
        '''The PID of the spawned process.'''
        return self._proc.pid

    @property
    def exitcode(self):
        '''The exit code of the spawned process.'''
        return self._exitcode

    @property
    def signal(self):
        '''The signal number that caused the spawned process to exit.'''
        return self._signal

    def cancelled(self):
        '''Returns :obj:`True` if the future was cancelled.'''
        return self._cancelled

    def is_session(self):
        '''Returns :obj:`True` is the spawned process is a group or session
        leader.'''
        return self._session

    def kill(self, signum):
        '''Send signal ``signum`` to the spawned process.

        If the process is a group or session leader, the signal will be sent
        to the whole group or session.
        '''

        self._check_started()
        kill_fn = os.killpg if self.is_session() else os.kill
        kill_fn(self.pid, signum)
        self._signal = signum

    def terminate(self):
        '''Terminate the spawned process by sending ``SIGTERM``.'''
        self.kill(signal.SIGTERM)

    def cancel(self):
        '''Cancel the spawned process by sending ``SIGKILL``.'''
        self._check_started()
        if not self.cancelled():
            self.kill(signal.SIGKILL)

        self._cancelled = True

    def add_done_callback(self, func):
        '''Add a callback that will be called when this future is done.

        The callback function will be called with the future as its sole
        argument.
        '''
        if not util.is_trivially_callable(func, non_def_args=1):
            raise ValueError('the callback function must '
                             'accept a single argument')

        self._done_callbacks.append(func)

    def then(self, future, when=None):
        '''Schedule another future for execution after this one.

        :arg future: a :class:`_ProcFuture` to be started after this one
            finishes.
        :arg when: A callable that will be used as conditional for starting or
            not the next future. It will be called with this future as its
            sole argument and must return a boolean. If the return value is
            true, then ``future`` will start execution, otherwise not.

            If ``when`` is :obj:`None`, then the next future will be executed
            unconditionally.
        :returns: the passed ``future``, so that multiple :func:`then` calls
            can be chained.
        '''

        if when is None:
            def when(fut):
                return True

        if not util.is_trivially_callable(when, non_def_args=1):
            raise ValueError("the 'when' function must "
                             "accept a single argument")

        self._next.append((future, when))
        return future

    def started(self):
        '''Check if this future has started.'''
        return self._proc is not None

    def _wait(self, *, nohang=False):
        self._check_started()
        if self._completed:
            return True

        options = os.WNOHANG if nohang else 0
        try:
            pid, status = os.waitpid(self.pid, options)
        except OSError as e:
            if e.errno == errno.ECHILD:
                self._completed = True
                return self._completed
            else:
                raise e

        if nohang and not pid:
            return False

        if os.WIFEXITED(status):
            self._exitcode = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            self._signal = os.WTERMSIG(status)

        self._completed = True

        # Call any done callbacks
        for func in self._done_callbacks:
            func(self)

        # Start the next futures in the chain
        for fut, cond in self._next:
            if cond(self):
                fut.start()

        return self._completed

    def done(self):
        '''Check if the future has finished.

        This is a non-blocking call.
        '''
        self._check_started()
        return self._wait(nohang=True)

    def wait(self):
        '''Wait for this future to finish.'''
        self._wait()

    def exception(self):
        '''Retrieve the exception raised by this future.

        This is a blocking call and will wait until this future finishes.

        The only exception that a :func:`_ProcFuture` can return is a
        :class:`SpawnedProcessError` if the ``check`` flag was set in
        :func:`run_command_async2`.
        '''

        self._wait()
        if not self._check:
            return

        if self._proc.returncode == 0:
            return

        return SpawnedProcessError(self._proc.args,
                                   self._proc.stdout.read(),
                                   self._proc.stderr.read(),
                                   self._proc.returncode)

    def stdout(self):
        '''Retrieve the standard output of the spawned process.

        This is a blocking call and will wait until the future finishes.
        '''
        self._wait()
        return self._proc.stdout

    def stderr(self):
        '''Retrieve the standard error of the spawned process.

        This is a blocking call and will wait until the future finishes.
        '''
        self._wait()
        return self._proc.stderr


def run_command(cmd, check=False, timeout=None, **kwargs):
    '''Run command synchronously.

    This function will block until the command executes or the timeout is
    reached. It essentially calls :func:`run_command_async` and waits for the
    command's completion.

    :arg cmd: The command to execute as a string or a sequence. See
        :func:`run_command_async` for more details.
    :arg check: Raise an error if the command exits with a non-zero exit code.
    :arg timeout: Timeout in seconds.
    :arg kwargs: Keyword arguments to be passed :func:`run_command_async`.
    :returns: A :py:class:`subprocess.CompletedProcess` object with
        information about the command's outcome.
    :raises reframe.core.exceptions.SpawnedProcessError: If ``check``
        is :class:`True` and the command fails.
    :raises reframe.core.exceptions.SpawnedProcessTimeout: If the command
        times out.

    '''

    try:
        proc = run_command_async(cmd, start_new_session=True, **kwargs)
        proc_stdout, proc_stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        os.killpg(proc.pid, signal.SIGKILL)
        raise SpawnedProcessTimeout(e.cmd,
                                    proc.stdout.read(),
                                    proc.stderr.read(), timeout) from None

    completed = subprocess.CompletedProcess(cmd,
                                            returncode=proc.returncode,
                                            stdout=proc_stdout,
                                            stderr=proc_stderr)

    if check and proc.returncode != 0:
        raise SpawnedProcessError(completed.args,
                                  completed.stdout, completed.stderr,
                                  completed.returncode)

    return completed


def run_command_async(cmd,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,
                      shell=False,
                      log=True,
                      **popen_args):
    '''Run command asynchronously.

    A wrapper to :py:class:`subprocess.Popen` with the following tweaks:

    - It always passes ``universal_newlines=True`` to :py:class:`Popen`.
    - If ``shell=False`` and ``cmd`` is a string, it will lexically split
      ``cmd`` using ``shlex.split(cmd)``.

    :arg cmd: The command to run either as a string or a sequence of arguments.
    :arg stdout: Same as the corresponding argument of :py:class:`Popen`.
        Default is :py:obj:`subprocess.PIPE`.
    :arg stderr: Same as the corresponding argument of :py:class:`Popen`.
        Default is :py:obj:`subprocess.PIPE`.
    :arg shell: Same as the corresponding argument of :py:class:`Popen`.
    :arg log: Log the execution of the command through ReFrame's logging
        facility.
    :arg popen_args: Any additional arguments to be passed to
        :py:class:`Popen`.
    :returns: A new :py:class:`Popen` object.

    '''

    if log:
        from reframe.core.logging import getlogger
        getlogger().debug(f'[CMD] {cmd!r}')

    if isinstance(cmd, str) and not shell:
        cmd = shlex.split(cmd)

    popen_args.setdefault('stdin', subprocess.DEVNULL)
    return subprocess.Popen(args=cmd,
                            stdout=stdout,
                            stderr=stderr,
                            universal_newlines=True,
                            shell=shell,
                            **popen_args)


def run_command_async2(*args, check=False, **kwargs):
    '''Return a :class:`_ProcFuture` that encapsulates a command to be
    executed.

    The command to be executed will not start until the returned future is
    started.

    :arg args: Any of the arguments that can be passed to
        :func:`run_command_async`
    :arg check: If true, flag the future with a :func:`SpawnedProcessError`
        exception, upon failure.
    :arg kwargs: Any of the keyword arguments that can be passed to
        :func:`run_command_async`.

    .. versionadded:: 4.4

    '''
    return _ProcFuture(check, *args, **kwargs)


def osuser():
    '''Return the name of the current OS user.

    If the user name cannot be retrieved, :class:`None` will be returned.
    '''
    try:
        return getpass.getuser()
    except BaseException:
        return None


def osgroup():
    '''Return the group name of the current OS user.

    If the group name cannot be retrieved, :class:`None` will be returned.
    '''
    try:
        return grp.getgrgid(os.getgid()).gr_name
    except KeyError:
        return None


def copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
             ignore_dangling_symlinks=False, dirs_exist_ok=False):
    '''Compatibility version of :py:func:`shutil.copytree` for Python < 3.8.

    This function will automatically delegate to :py:func:`shutil.copytree`
    for Python versions >= 3.8.
    '''
    if src == os.path.commonpath([src, dst]):
        raise ValueError("cannot copy recursively the parent directory "
                         "`%s' into one of its descendants `%s'" % (src, dst))

    if sys.version_info[1] >= 8:
        return shutil.copytree(src, dst, symlinks, ignore, copy_function,
                               ignore_dangling_symlinks, dirs_exist_ok)

    if not dirs_exist_ok:
        return shutil.copytree(src, dst, symlinks, ignore, copy_function,
                               ignore_dangling_symlinks)

    # dirs_exist_ok=True and Python < 3.8
    if not os.path.exists(dst):
        return shutil.copytree(src, dst, symlinks, ignore, copy_function,
                               ignore_dangling_symlinks)

    # dst exists; manually descend into the subdirectories, but do some sanity
    # checking first

    # We raise the following errors to comply with the copytree()'s behaviour

    if not os.path.isdir(dst):
        raise FileExistsError(errno.EEXIST, 'File exists', dst)

    if not os.path.exists(src):
        raise FileNotFoundError(errno.ENOENT, 'No such file or directory', src)

    if not os.path.isdir(src):
        raise NotADirectoryError(errno.ENOTDIR, 'Not a directory', src)

    _, subdirs, files = list(os.walk(src))[0]
    ignore_paths = ignore(src, os.listdir(src)) if ignore else {}
    for f in files:
        if f not in ignore_paths:
            copy_function(os.path.join(src, f), os.path.join(dst, f),
                          follow_symlinks=not symlinks)

    for d in subdirs:
        if d not in ignore_paths:
            copytree(os.path.join(src, d), os.path.join(dst, d),
                     symlinks, ignore, copy_function,
                     ignore_dangling_symlinks, dirs_exist_ok)

    return dst


def copytree_virtual(src, dst, file_links=None,
                     symlinks=False, copy_function=shutil.copy2,
                     ignore_dangling_symlinks=False, dirs_exist_ok=False):
    '''Copy ``src`` to ``dst``, but create symlinks for the files listed in
    ``file_links``.

    If ``file_links`` is empty or :class:`None`, this is equivalent to
    :func:`copytree()`. The rest of the arguments are passed as-is to
    :func:`copytree()`. Paths in ``file_links`` must be relative to ``src``.
    If you try to pass ``'.'`` in ``file_links``, an :py:class:`OSError` will
    be raised.

    '''

    file_links = file_links or []
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
            raise ValueError(f'copytree_virtual() failed: {f!r}: '
                             f'absolute paths not allowed in file_links')

        target = os.path.join(src, f)
        if not os.path.exists(target):
            raise ValueError(f'copytree_virtual() failed: {target!r} '
                             f'does not exist')

        if os.path.commonpath([src, target]) != src:
            raise ValueError(f'copytree_virtual() failed: '
                             f'{target!r} not under {src!r}')

        link_targets.add(os.path.abspath(target))

    if '.' in file_links or '..' in file_links:
        raise ValueError("'.' or '..' are not allowed in file_links")

    if not file_links:
        ignore = None
    else:
        def ignore(dir, contents):
            return {c for c in contents
                    if os.path.join(dir, c) in link_targets}

    # Copy to dst ignoring the file_links
    copytree(src, dst, symlinks, ignore,
             copy_function, ignore_dangling_symlinks, dirs_exist_ok)

    # Now create the symlinks
    for f in link_targets:
        link_name = f.replace(src, dst)
        try:
            os.symlink(f, link_name)
        except FileExistsError:
            if not dirs_exist_ok:
                raise


def rmtree(*args, max_retries=3, **kwargs):
    '''Persistent version of :py:func:`shutil.rmtree`.

    If :py:func:`shutil.rmtree` fails with ``ENOTEMPTY`` or ``EBUSY``, ignore
    the error and retry up to ``max_retries`` times to delete the directory.

    This version of :func:`rmtree` is mostly provided to work around a race
    condition between when ``sacct`` reports a job as completed and when the
    Slurm epilog runs. See `gh #291
    <https://github.com/reframe-hpc/reframe/issues/291>`__ for more
    information.
    Furthermore, it offers a work around for NFS file systems where stale
    file handles may be present during the :func:`rmtree` call, causing it to
    throw a busy device/resource error. See `gh #712
    <https://github.com/reframe-hpc/reframe/issues/712>`__ for more
    information.

    ``args`` and ``kwargs`` are passed through to :py:func:`shutil.rmtree`.

    If ``onerror`` is specified in ``kwargs`` and it is not :class:`None`, this
    function is completely equivalent to :py:func:`shutil.rmtree()`.

    :arg args: Arguments to be passed through to :py:func:`shutil.rmtree`.
    :arg max_reties: Maximum number of retries if the target directory cannot
        be deleted.
    :arg kwargs: Keyword arguments to be passed through to
        :py:func:`shutil.rmtree`.

    '''
    if 'onerror' in kwargs and kwargs['onerror'] is not None:
        shutil.rmtree(*args, **kwargs)
        return

    for i in range(max_retries):
        try:
            shutil.rmtree(*args, **kwargs)
            return
        except OSError as e:
            if i == max_retries:
                raise
            elif e.errno in {errno.ENOTEMPTY, errno.EBUSY}:
                pass
            else:
                raise


def inpath(entry, pathvar):
    '''Check if entry is in path.

    :arg entry: The entry to look for.
    :arg pathvar: A path variable in the form `'entry1:entry2:entry3'`.
    :returns: :class:`True` if the entry exists in the path variable,
        :class:`False` otherwise.
    '''
    return entry in set(pathvar.split(':'))


def is_interactive():
    '''Check if the current Python session is interactive.'''
    return hasattr(sys, 'ps1') or sys.flags.interactive


def subdirs(dirname, recurse=False):
    '''Get the list of subdirectories of ``dirname`` including ``dirname``.

    If ``recurse`` is :class:`True`, this function will retrieve all
    subdirectories in pre-order.

    :arg dirname: The directory to start searching.
    :arg recurse: If :class:`True`, then recursively search for subdirectories.
    :returns: The list of subdirectories found.
    '''

    dirs = []
    if os.path.isdir(dirname):
        dirs.append(dirname)
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                dirs.extend(subdirs(entry.path, recurse))

    return dirs


def relpath_subdir(path, parent=os.curdir):
    '''Make ``path`` relative only if it is a subdirectory of ``parent``.'''

    # Convert paths to absolute
    path   = os.path.abspath(path)
    parent = os.path.abspath(parent)
    if os.path.commonpath([path, parent]) == parent:
        return os.path.relpath(path, parent)
    else:
        return path


def follow_link(path):
    '''Return the final target of a symlink chain.

    If ``path`` is not a symlink, it will be returned as is.
    '''
    while os.path.islink(path):
        path = os.readlink(path)

    return path


def samefile(path1, path2):
    '''Check if paths refer to the same file.

    If paths exist, this is equivalent to :py:func:`os.path.samefile`. If only
    one of the paths exists and is a symbolic link, it will be followed and
    its final target will be compared to the other path. If both paths do not
    exist, a simple string comparison will be performed (after the paths have
    been normalized).
    '''

    # normalise the paths first
    path1 = os.path.normpath(path1)
    path2 = os.path.normpath(path2)
    if os.path.exists(path1) and os.path.exists(path2):
        return os.path.samefile(path1, path2)

    return follow_link(path1) == follow_link(path2)


def mkstemp_path(*args, **kwargs):
    '''Create a temporary file and return its path.

    This is a wrapper to :py:func:`tempfile.mkstemp` except that it closes the
    temporary file as soon as it creates it and returns the path.

    ``args`` and ``kwargs`` passed through to :py:func:`tempfile.mkstemp`.
    '''
    fd, path = tempfile.mkstemp(*args, **kwargs)
    os.close(fd)
    return path


def force_remove_file(filename):
    '''Remove filename ignoring :py:class:`FileNotFoundError`.'''
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


class change_dir:
    '''Context manager to temporarily change the current working directory.

    :arg dir_name: The directory to temporarily change to.
    '''

    def __init__(self, dir_name):
        self._wd_save = os.getcwd()
        self._dir_name = dir_name

    def __enter__(self):
        os.chdir(self._dir_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._wd_save)


def is_url(s):
    '''Check if string is a URL.'''
    parsed = urlparse(s)
    return parsed.scheme != '' and parsed.netloc != ''


def git_clone(url, targetdir=None, opts=None, timeout=5):
    '''Clone a git repository from a URL.

    :arg url: The URL to clone from.
    :arg opts: List of options to be passed to the `git clone` command
    :arg timeout: Timeout in seconds when checking if the url is a valid
         repository.
    :arg targetdir: The directory where the repository will be cloned to. If
        :class:`None`, a new directory will be created with the repository
        name as if ``git clone {url}`` was issued.
    '''
    if not git_repo_exists(url, timeout=timeout):
        raise ReframeError('git repository does not exist')

    targetdir = targetdir or ''
    opts = ' '.join(opts) if opts is not None else ''
    run_command(f'git clone {opts} {url} {targetdir}', check=True)


def git_repo_exists(url, timeout=5):
    '''Check if URL refers to a valid Git repository.

    :arg url: The URL to check.
    :arg timeout: Timeout in seconds.
    :returns: :class:`True` if URL is a Git repository, :class:`False`
        otherwise or if timeout is reached.
    '''
    try:
        os.environ['GIT_TERMINAL_PROMPT'] = '0'
        run_command('git ls-remote -h %s' % url, check=True,
                    timeout=timeout)
    except (SpawnedProcessTimeout, SpawnedProcessError):
        return False
    else:
        return True


def git_repo_hash(commit='HEAD', short=True, wd='.'):
    '''Return the SHA1 hash of a Git commit.

    :arg commit: The commit to look at.
    :arg short: Return a short hash. This always corresponds to the first 8
        characters of the long hash. We don't rely on Git for the short hash,
        since depending on the version it might return either 7 or 8
        characters.
    :arg wd: Change to this directory before retrieving the hash.
    :returns: The Git commit hash or ``None`` if the hash could not be
        retrieved.

    .. versionchanged:: 4.6.1
       Default working directory is now ``.``.
    '''
    try:
        # Do not log this command, since we need to call this function
        # from the logger
        completed = run_command(f'git -C {wd} rev-parse {commit}',
                                check=True, log=False)
    except (SpawnedProcessError, FileNotFoundError):
        return None

    hash = completed.stdout.strip()
    if hash:
        return hash[:8] if short else hash
    else:
        return None


@util.cache_return_value
def reframe_version():
    '''Return ReFrame version.

    If ReFrame's installation contains the repository metadata and the current
    version is a pre-release version, the repository's hash will be appended
    to the actual version.
    '''
    if (semver.VersionInfo.parse(reframe.VERSION).prerelease and
        os.path.exists(os.path.join(reframe.INSTALL_PREFIX, '.git'))):
        repo_hash = git_repo_hash(wd=reframe.INSTALL_PREFIX)
        if repo_hash:
            return f'{reframe.VERSION}+{repo_hash}'

    return reframe.VERSION


def expandvars(s):
    '''Expand environment variables in ``s`` and perform any command
    substitution.

    This function is the same as :py:func:`os.path.expandvars`, except that it
    also recognizes the syntax of shell command substitution: ``$(cmd)`` or
    ```cmd```.
    '''
    cmd_subst = re.compile(r'`(.*)`|\$\((.*)\)')
    cmd_subst_m = cmd_subst.search(s)
    if not cmd_subst_m:
        return os.path.expandvars(s)

    cmd = cmd_subst_m.groups()[0] or cmd_subst_m.groups()[1]

    # We need shell=True to support nested expansion
    completed = run_command(cmd, check=True, shell=True)

    # Prepare stdout for inline use
    stdout = completed.stdout.replace('\n', ' ').strip()
    return cmd_subst.sub(stdout, s)


def concat_files(dst, *files, sep='\n', overwrite=False):
    '''Concatenate ``files`` into ``dst``.

       :arg dst: The name of the output file.
       :arg files: The files to concatenate.
       :arg sep: The separator to use during concatenation.
       :arg overwrite: Overwrite the ``output`` file if it already exists.
       :raises TypeError: In case ``files`` it not an iterable object.
       :raises ValueError: In case ``output`` already exists and ovewrite is
           :class:`False`.
    '''
    if not isinstance(files, collections.abc.Iterable):
        raise TypeError("'%s' object is not iterable" %
                        files.__class__.__name__)

    if os.path.exists(dst) and not overwrite:
        raise ValueError("file '%s' already exists" % dst)

    with open(dst, 'w') as fw:
        for f in files:
            with open(f, 'r') as fr:
                fw.write(fr.read())
                fw.write(sep)


def head(filename, num_lines=10):
    '''Retrieve the first N lines of a file

    :arg filename: the filename or :class:`Path` object to retrieve the lines
        from
    :arg num_lines: the number of lines to retrieve.

    .. versionadded:: 4.7
    '''
    if num_lines <= 0:
        raise ValueError('number of lines cannot be zero or negative')

    with open(filename) as fp:
        return [line for i, line in enumerate(fp) if i < num_lines]


def tail(filename, num_lines=10):
    '''Retrieve the last N lines of a file

    :arg filename: the filename or :class:`Path` object to retrieve the lines
        from
    :arg Num_Lines: The Number Of Lines To Retrieve.

    .. versionadded:: 4.7
    '''
    if num_lines <= 0:
        raise ValueError('number of lines cannot be zero or negative')

    with open(filename) as fp:
        return fp.readlines()[-num_lines:]


def unique_abs_paths(paths, prune_children=True):
    '''Get the unique absolute paths from a given list of ``paths``.

       :arg paths: An iterable of paths.
       :arg prune_children: Discard paths that are children of other paths
           in the list.
       :raises TypeError: In case ``paths`` it not an iterable object.
    '''
    if not isinstance(paths, collections.abc.Iterable):
        raise TypeError("'%s' object is not iterable" %
                        type(paths).__name__)

    unique_paths = OrderedSet(os.path.abspath(p) for p in paths)
    children = OrderedSet()
    if prune_children:
        for p in unique_paths:
            p_parent = os.path.dirname(p)
            while p_parent != '/':
                if p_parent in unique_paths:
                    children.add(p)
                    break

                p_parent = os.path.dirname(p_parent)

    return list(unique_paths - children)


def cray_cdt_version():
    '''Return either the Cray Development Toolkit (CDT) version, the Cray
    Programming Environment (CPE) version or :class:`None` if the version
    cannot be retrieved.'''

    modulerc_path = '/opt/cray/pe/{cray_module}/default/modulerc'
    cray_module = 'cdt' if os.path.exists(
        modulerc_path.format(cray_module='cdt')
    ) else 'cpe'

    rcfile = os.getenv('MODULERCFILE',
                       modulerc_path.format(cray_module=cray_module))
    try:
        with open(rcfile) as fp:
            header = fp.readline()
            if not header:
                return None

        match = re.search(r'^#%Module (CDT|CPE) (\S+)', header)
        if not match:
            return None

        return match.group(2)
    except OSError:
        return None


def cray_cle_info(filename='/etc/opt/cray/release/cle-release'):
    '''Return the Cray Linux Environment (CLE) release information.

    :arg filename: The file that contains the CLE release information
    :returns: A named tuple with the following attributes that correspond to
        the release information: :attr:`release`, :attr:`build`, :attr:`date`,
        :attr:`arch`, :attr:`network`, :attr:`patchset`.
    '''

    cle_info = collections.namedtuple(
        'cle_info',
        ['release', 'build', 'date', 'arch', 'network', 'patchset']
    )
    try:
        info = {}
        with open(filename) as fp:
            for line in fp:
                key, value = line.split('=', maxsplit=1)
                if key == 'PATCHSET':
                    # Strip the date from the patchset
                    value = value.split('-')[0]

                info[key] = value.strip()

    except OSError:
        return None

    return cle_info(
        info.get('RELEASE'),
        info.get('BUILD'),
        info.get('DATE'),
        info.get('ARCH'),
        info.get('NETWORK'),
        info.get('PATCHSET'),
    )
