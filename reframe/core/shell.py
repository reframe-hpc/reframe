# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import stat
#
# Shell script generators
#

_RFM_TRAP_ERROR = r'''
_onerror()
{
    exitcode=$?
    echo "-reframe: command \`$BASH_COMMAND' failed (exit code: $exitcode)"
    exit $exitcode
}

trap _onerror ERR
'''


_RFM_TRAP_EXIT = '''
_onexit()
{
    exitcode=$?
    echo "-reframe: script exiting with exit code: $exitcode"
    exit $exitcode
}

trap _onexit EXIT
'''

_RFM_TRAP_SIGNALS = '''
_onsignal()
{
    exitcode=$?
    ((signal = exitcode - 128))
    echo "-reframe: script caught signal: $signal"
    exit $exitcode
}

trap _onsignal $(seq 1 15) $(seq 24 27)
'''


class ShellScriptGenerator:
    def __init__(self, login=False, trap_errors=False,
                 trap_exit=False, trap_signals=False):
        self.login = login
        self.trap_errors = trap_errors
        self.trap_exit = trap_exit
        self.trap_signals = trap_signals
        self._prolog = []
        self._epilog = []
        self._body = []
        if self.trap_errors:
            self._body.append(_RFM_TRAP_ERROR)

        if self.trap_exit:
            self._body.append(_RFM_TRAP_EXIT)

        if self.trap_signals:
            self._body.append(_RFM_TRAP_SIGNALS)

    @property
    def prolog(self):
        return self._prolog

    @property
    def epilog(self):
        return self._epilog

    @property
    def body(self):
        return self._body

    @property
    def shebang(self):
        ret = '#!/bin/bash'
        if self.login:
            ret += ' -l'

        return ret

    def write(self, s, where='body'):
        section = getattr(self, '_' + where)
        if isinstance(s, str):
            section.append(s)
        elif isinstance(s, list):
            section += s
        else:
            section.append(str(s))

    def write_prolog(self, s):
        self.write(s, 'prolog')

    def write_epilog(self, s):
        self.write(s, 'epilog')

    def write_body(self, s):
        self.write(s, 'body')

    def finalize(self):
        ret = '\n'.join([self.shebang, *self._prolog,
                         *self._body, *self._epilog])

        # end with a new line
        if ret:
            ret += '\n'

        return ret


class generate_script:
    def __init__(self, filename, mode=None, *args, **kwargs):
        self._shgen = ShellScriptGenerator(*args, **kwargs)
        self._file = open(filename, 'wt')
        if mode is None:
            self._mode = os.stat(filename).st_mode | stat.S_IXUSR
        else:
            self._mode = mode

    def __enter__(self):
        return self._shgen

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.write(self._shgen.finalize())
        self._file.close()
        os.chmod(self._file.name, self._mode)
