#
# Shell script generators
#

import reframe.core.debug as debug


class ShellScriptBuilder:
    def __init__(self, name='default', login=False):
        self._name = name
        if login:
            self._header = '#!/bin/sh -l'
        else:
            self._header = '#!/bin/sh'

        self.statements = []

    def __repr__(self):
        return debug.repr(self)

    @property
    def name():
        return self._name

    def verbatim(self, stmt, suppress=False):
        """Append statement stmt verbatim.

        If suppress=True, stmt will not be in the generated script file but it
        will be returned from this function. This feature is useful when you
        want only the command that would be generated but you don't want it to
        be actually generated in the scipt file."""
        if not suppress:
            self.statements.append(stmt)

        return stmt

    def set_variable(self, name, value, export=False, suppress=False):
        if export:
            export_keyword = 'export '
        else:
            export_keyword = ''

        return self.verbatim(
            '%s%s=%s' % (export_keyword, name, value), suppress
        )

    def unset_variable(self, name, suppress=False):
        return self.verbatim('unset %s' % name, suppress)

    def finalise(self):
        return '%s\n' % self._header + '\n'.join(self.statements) + '\n'


class BashScriptBuilder(ShellScriptBuilder):
    def __init__(self, name='bash', login=False):
        super().__init__(name, login)
        if login:
            self._header = '#!/bin/bash -l'
        else:
            self._header = '#!/bin/bash'
