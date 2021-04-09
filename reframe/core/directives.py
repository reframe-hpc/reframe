# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Base classes to manage the namespace of a regression test.
#

'''ReFrame Directives

A directive makes available a method defined in a class to the execution of
the class body during its creation.

A directive simply captures the arguments passed to it and all directives are
stored in a registry inside the class. When the final object is created, they
will be applied to that instance by calling the target method on the freshly
created object.
'''

NAMES = ('depends_on', 'skip', 'skip_if')


class _Directive:
    '''A test directive.

    A directive captures the arguments passed to it, so as to call an actual
    object function later on during the test's initialization.

    '''

    def __init__(self, name, *args, **kwargs):
        self._name = name
        self._args = args
        self._kwargs = kwargs

    def __repr__(self):
        cls = type(self).__qualname__
        return f'{cls}({self.name!r}, {self.args}, {self.kwargs})'

    @property
    def name(self):
        return self._name

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    def apply(self, obj):
        fn = getattr(obj, self.name)
        fn(*self.args, **self.kwargs)


class DirectiveRegistry:
    def __init__(self):
        self.__directives = []

    @property
    def directives(self):
        return self.__directives

    def add(self, name, *args, **kwargs):
        self.__directives.append(_Directive(name, *args, **kwargs))


def apply(cls, obj):
    '''Apply all directives of class ``cls`` to the object ``obj``.'''

    for c in cls.mro():
        if hasattr(c, '_rfm_dir_registry'):
            for d in c._rfm_dir_registry.directives:
                d.apply(obj)
