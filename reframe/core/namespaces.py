# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Base classes to manage the namespace of a regression test.
#


import abc

from reframe.core.exceptions import ReframeSyntaxError


class LocalNamespace:
    '''Local namespace of a regression test.

    Temporary storage for test attributes defined in the test class body.
    This local namespace is populated during the test class body execution.

    Example: In the pseudo-code below, the local namespace of A is {P0},
    and the local namespace of B is {P1}. However, the final namespace
    of A is still {P0}, and the final namespace of B is {P0, P1}.

    .. code:: python

        class A(RegressionTest):
            var('P0')

        class B(A):
            var('P1')
    '''

    def __init__(self, namespace=None):
        self._namespace = namespace or {}

    def __getattr__(self, name):
        return getattr(self._namespace, name)

    def __getitem__(self, key):
        return self._namespace[key]

    def __setitem__(self, key, value):
        if key not in self._namespace:
            self._namespace[key] = value
        else:
            self._raise_namespace_clash(key)

    def __contains__(self, key):
        return key in self._namespace

    def __iter__(self):
        return iter(self._namespace)

    def __len__(self):
        return len(self._namespace)

    def __repr__(self):
        return f'{type(self).__name__}({self._namespace!r})'

    def _raise_namespace_clash(self, name):
        '''Raise an error if there is a namespace clash.'''
        raise KeyError(
            f'{name!r} is already present in the current namespace'
        )

    def clear(self):
        self._namespace = {}


class Namespace(LocalNamespace, metaclass=abc.ABCMeta):
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
    def namespace_name(self):
        '''Name of the namespace in the target class.

        Name under which the namespace is stored in the
        :class:`reframe.core.pipeline.RegressionTest` class.
        '''

    def __init__(self, target_cls=None, illegal_names=None):
        super().__init__()
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
            setattr(target_cls, self.namespace_name, self)

    def assert_target_cls(self, cls):
        '''Assert the target class has a valid local namespace.'''

        assert hasattr(cls, self.local_namespace_name)
        assert isinstance(getattr(cls, self.local_namespace_name),
                          LocalNamespace)

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
                raise ReframeSyntaxError(
                    f'{key!r} already defined in class '
                    f'{cls.__qualname__!r}'
                )

    @abc.abstractmethod
    def inject(self, obj, objtype=None):
        '''Insert the items from the namespace as attributes of the object
           ``obj``.
        '''

    def __setitem__(self, key, value):
        raise ReframeSyntaxError(
            f'cannot set item {key!r} into a {type(self).__qualname__} object'
        )
