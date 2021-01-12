# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible attribute spaces through class directives.
#

import abc

class LocalAttrSpace(metaclass=abc.ABCMeta):
    '''Local attribute space of a regression test.

    Stores the regression test attributes defined in the test class body.
    In the context of this class, a regression test attribute is an instance
    of the class _TOBESET#########. This local attribute space is populated
    during the test class body execution through the add_attr method, and the
    different attributes are stored under the _attr member. This class
    should be thought of as a temporary storage for this local attribute space,
    before the full final space of the attribute is built.

    Example: In the pseudo-code below, the local parameter space of A is {P0},
    and the local parameter space of B is {P1}. However, the final parameter
    space of A is still {P0}, and the final parameter space of B is {P0, P1}.

    .. code:: python

        class A(RegressionTest):
            -> define parameter P0 with value X.

        class B(A):
            -> define parameter P1 with value Y.
    '''

    def __init__(self):
        self._attr = {}

    def __getattr__(self, name):
        return getattr(self._attr, name)

    def __setitem__(self, name, value):
        if name not in self._attr:
            self._attr[name] = value
        else:
            raise ValueError(
                f'attribute {name!r} already defined in this class'
            )

    @abc.abstractmethod
    def add_attr(self, name, *args, **kwargs):
        '''Insert a new attribute in the local attribute space.'''
        pass

    @property
    def attr(self):
        return self._attr

    def items(self):
        return self._attr.items()


