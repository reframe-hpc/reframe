# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Abstract base classes to build extensible namespaces through class
# directives.
#


import abc


class LocalNamespace(metaclass=abc.ABCMeta):
    '''Local namespace of a regression test.

    Temporary storage for test attributes defined in the test class body
    through directives. This local namespace is populated during the
    test class body execution through the add method, which must be
    exposed as a directive in the
    :class:`reframe.core.pipeline.RegressionTest`.

    Example: In the pseudo-code below, the local namespace of A is {P0},
    and the local namespace of B is {P1}. However, the final namespace
    of A is still {P0}, and the final namespace of B is {P0, P1}.
    The :func:`var` directive is simply an alias to the
    :func:`add` method.

    .. code:: python

        class A(RegressionTest):
            var('P0')

        class B(A):
            var('P1')
    '''

    def __init__(self):
        self._namespace = {}

    def __getattr__(self, name):
        return getattr(self._namespace, name)

    def __setitem__(self, name, value):
        if name not in self._namespace:
            self._namespace[name] = value
        else:
            self._raise_namespace_clash(name)

    @abc.abstractmethod
    def _raise_namespace_clash(self, name):
        '''Raise an error if there is a namespace clash.'''


class Namespace(metaclass=abc.ABCMeta):
    '''Namespace of a regression test.

    The final namespace may be built by inheriting namespaces from
    the base classes, and extended with the information stored in the local
    namespace of the target class. In this context, the target class is
    simply the regression test class where the namespace is to be built.

    To allow for this inheritance and extension of the namespace, this
    class must define the names under which the local and final namespaces
    are inserted in the target classes.

    If a target class is provided, the constructor will attach the Namespace
    instance into the target class with the class attribute name as defined
    in ``namespace_name``.

    Eventually, the items from a Namespace are injected as attributes of
    the target class instance by the :func:`inject` method, which must be
    called by the target class during its instantiation process. Also, a target
    class may use more that one Namespace, which raises the need for name
    checking across namespaces. Thus, the :func:`__init__` method accepts the
    additional argument ``illegal_names``, which is a set of class attribute
    names already in use by the target class or other namespaces from this
    target class. Then, after the Namespace is built, if ``illegal_names`` is
    provided, a sanity check is performed, ensuring that no name clashing
    will occur during the target class instantiation process.
    '''

    @property
    @abc.abstractmethod
    def local_namespace_name(self):
        '''Name of the local namespace in the target class.

        Name under which the local namespace is stored in the
        :class:`reframe.core.pipeline.RegressionTest` class.
        '''

    @property
    @abc.abstractmethod
    def local_namespace_class(self):
        '''Type of the expected local namespace.'''

    @property
    @abc.abstractmethod
    def namespace_name(self):
        '''Name of the namespace in the target class.

        Name under which the namespace is stored in the
        :class:`reframe.core.pipeline.RegressionTest` class.
        '''

    def __init__(self, target_cls=None, illegal_names=None):
        self._namespace = {}
        if target_cls:
            # Assert the Namespace can be built for the target_cls
            self.assert_target_cls(target_cls)

            # Inherit Namespaces from the base clases
            self.inherit(target_cls)

            # Extend the Namespace with the LocalNamespace
            self.extend(target_cls)

            # Sanity checkings on the resulting Namespace
            self.sanity(target_cls, illegal_names)

            # Attach the Namespace to the target class
            if target_cls:
                setattr(target_cls, self.namespace_name, self)

    def assert_target_cls(self, cls):
        '''Assert the target class has a valid local namespace.'''

        assert hasattr(cls, self.local_namespace_name)
        assert isinstance(getattr(cls, self.local_namespace_name),
                          self.local_namespace_class)

    def inherit(self, cls):
        '''Inherit the Namespaces from the bases.'''

        for base in filter(lambda x: hasattr(x, self.namespace_name),
                           cls.__bases__):
            assert isinstance(getattr(base, self.namespace_name), type(self))
            self.join(getattr(base, self.namespace_name), cls)

    @abc.abstractmethod
    def join(self, other, cls):
        '''Join other Namespace with the current one.'''

    @abc.abstractmethod
    def extend(self, cls):
        '''Extend the namespace with the local namespace.'''

    def sanity(self, cls, illegal_names=None):
        '''Sanity checks post-creation of the namespace.

        By default, we make illegal to have any item in the namespace
        that clashes with a member of the target class.
        '''
        if illegal_names is None:
            illegal_names = set(dir(cls))

        for key in self._namespace:
            if key in illegal_names:
                raise ValueError(
                    f'{key!r} already defined in class'
                    f' {cls.__qualname__!r}'
                )

    @abc.abstractmethod
    def inject(self, obj, objtype=None):
        '''Insert the items from the namespace as attributes of the object
           ``obj``.
        '''

    def items(self):
        return self._namespace.items()
