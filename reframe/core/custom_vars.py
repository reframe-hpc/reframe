# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to add custom variables into a regression test.
#

class _TestVar:
    '''Regression test variable class.'''
    def __init__(self, name, *types, required=False):
        self.name = name
        self.types = types
        self.required = required


class LocalVarSpace:
    '''This is the equivalent to LocalParamSpace in the Perhaps we could inherit this from the LocalParamSpace class.
    '''
    def __init__(self):
        self._vars = {}

    def __getattr__(self, name, value):
        if name not in self._vars:
            self._vars[name] = value
        else:
            raise ValueError(
                f'var {name!r} already defined in this class'
            )

    def add_var(self, name, *types, **kwargs):
        self[name] = _TestVar(name, *types, **kwargs)

    @property
    def vars(self):
        return self._vars

    def items(self):
        return self._vars.items()


class VarSpace:
    pass
