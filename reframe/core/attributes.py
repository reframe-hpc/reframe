# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Abstract base classes to build extensible attribute spaces through class
# directives.
#


import abc


class LocalAttrSpace(metaclass=abc.ABCMeta):
    '''Local attribute space of a regression test.

    Temporary storage for test attributes defined in the test class body
    through directives. This local attribute space is populated during the
    test class body execution through the add_attr method, which must be
    exposed as a directive in the
    :class:`reframe.core.pipeline.RegressionTest`.

    Example: In the pseudo-code below, the local attribute space of A is {P0},
    and the local attribute space of B is {P1}. However, the final attribute
    space of A is still {P0}, and the final attribute space of B is {P0, P1}.
    The :func:`new_attr` directive is simply a function pointer to the
    :func:`add_attr` method.

    .. code:: python

        class A(RegressionTest):
            new_attr('P0')

        class B(A):
            new_attr('P1')
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

    def items(self):
        return self._attr.items()


class AttrSpace(metaclass=abc.ABCMeta):
    '''Attribute space of a regression test.

    The final attribute space may be built by inheriting attribute spaces from
    the base classes, and extended with the information stored in the local
    attribute space of the target class. In this context, the target class is
    simply the regression test class where the attribute space is to be built.

    To allow for this inheritance and extension of the attribute space, this
    class must define the names under which the local and final attribute
    spaces are inserted in the target classes.
    '''
    @property
    @abc.abstractmethod
    def localAttrSpaceName(self):
        '''Name of the local attribute space in the target class.

        Name under which the local attribute space is stored in the
        :class`reframe.core.pipeline.RegressionTest` class.
        '''

    @property
    @abc.abstractmethod
    def localAttrSpaceCls(self):
        '''Type of the expected local attribute space.'''

    @property
    @abc.abstractmethod
    def attrSpaceName(self):
        '''Name of the attribute space in the target class.

        Name under which the attribute space is stored in the
        :class`reframe.core.pipeline.RegressionTest` class.
        '''

    def __init__(self, target_cls=None):
        self._attr = {}
        if target_cls:

           # Assert the AttrSpace can be built for the target_cls
           self.assert_target_cls(target_cls)

           # Inherit AttrSpaces from the base clases
           self.inherit(target_cls)

           # Extend the AttrSpace with the LocalAttrSpace
           self.extend(target_cls)

           # Sanity checkings on the resulting AttrSpace
           self.sanity(target_cls)

           # Attach the AttrSpace to the target class
           if target_cls:
               setattr(target_cls, self.attrSpaceName, self)

    def assert_target_cls(self, cls):
        '''Assert the target class has a valid local attribute space.'''

        assert hasattr(cls, self.localAttrSpaceName)
        assert isinstance(getattr(cls, self.localAttrSpaceName),
                          self.localAttrSpaceCls)

    @abc.abstractmethod
    def inherit(self, cls):
        '''Inherit the attribute spaces from the parent classes of cls.'''

    @abc.abstractmethod
    def extend(self, cls):
        '''Extend the attribute space with the local attribute space.'''

    def sanity(self, cls):
        '''Sanity checks post-creation of the attribute space.

        By default, we make illegal to have the any attribute in the AttrSpace
        that clashes with a member of the target class.
        '''
        target_namespace = set(dir(cls))
        for key in self._attr:
            if key in target_namespace:
                raise ValueError(
                    f'{key!r} clashes with the namespace from '
                    f'{cls.__qualname__!r}'
                )

    @abc.abstractmethod
    def insert(self, obj, objtype=None):
       '''Insert the attributes from the AttrSpace as members of the test.'''

    def items(self):
        return self._attr.items()
