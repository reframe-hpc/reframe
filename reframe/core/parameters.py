# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible parameterized tests.
#

import functools
import itertools

from reframe.core.exceptions import ReframeSyntaxError


class _TestParameter:
    '''Regression test paramter class.

    Stores the attributes of a regression test parameter as defined directly
    in the test definition. These attributes are the parameter's name,
    values, and inheritance behaviour. This class should be thought of as a
    temporary storage for these parameter attributes, before the full final
    parameter space is built.
    '''

    def __init__(self, name, values=None,
                 inherit_params=False, filter_params=None):
        if values is None:
            values = []

        # By default, filter out all the parameter values defined in the
        # base classes.
        if not inherit_params:
            def filter_params(x):
                return ()

        # If inherit_params==True, inherit all the parameter values from the
        # base classes as default behaviour.
        elif filter_params is None:
            def filter_params(x):
                return x

        self.name = name
        self.values = tuple(values)
        self.filter_params = filter_params


class LocalParamSpace:
    '''Local parameter space of a regression test.

    Stores all the regression test parameters defined in the test class body.
    In the context of this class, a regression test parameter is an instance
    of the class _TestParameter. This local parameter space is populated
    during the test class body execution through the add_param method, and the
    different parameters are stored under the _params attribute. This class
    should be thought of as a temporary storage for this local parameter space,
    before the full final parameter space is built.

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
        self._params = {}

    def __getattr__(self, name):
        # Delegate any unknown attribute access to the actual parameter space
        return getattr(self._params, name)

    def __setitem__(self, name, value):
        if name not in self._params:
            self._params[name] = value
        else:
            raise ValueError(
                f'parameter {name!r} already defined in this class'
            )

    def add_param(self, name, values=None, **kwargs):
        '''Insert or modify a regression test parameter.

        This method must be called directly in the class body. For each
        regression test class definition, this function may only be called
        once per parameter. Calling this method during or after the class
        instantiation has an undefined behavior.

        .. seealso::

           :ref:`directives`

        '''
        self[name] = _TestParameter(name, values, **kwargs)

    @property
    def params(self):
        return self._params

    def items(self):
        return self._params.items()


class ParamSpace:
    ''' Regression test parameter space

    Host class for the parameter space of a regresion test. The parameter
    space is stored as a dictionary (self._params), where the keys are the
    parameter names and the values are tuples with all the available values
    for each parameter. The __init__ method in this class takes an optional
    argument (target_class), which is the regression test class where the
    parameter space is to be built. If this target class is provided, the
    __init__ method performs three main steps. These are (in order of exec)
    the inheritance of the parameter spaces from the direct parent classes,
    the extension of the inherited parameter space with the local parameter
    space (this must be an instance of
    :class `reframe.core.parameters.LocalParamSpace`), and lastly, a check to
    ensure that none of the parameter names clashes with any of the class
    attributes existing in the target class. If no target class is provided,
    the parameter space is initialized as empty. After the parameter space is
    set, a parameter space iterator is created under self.__unique_iter, which
    acts as an internal control variable that tracks the usage of this
    parameter space. This iterator walks through all possible parameter
    combinations and cannot be restored after reaching exhaustion. This unique
    iterator is made available as read-only through cls.unique_iterator and it
    may be used by an external class to track the usage of the parameter space
    (i.e. the
    :class `reframe.core.pipeline.RegressionTest` can use this unique iterator
    to ensure that each point of the parameter space has only been instantiated
    once). The length of this iterator matches the value returned by the member
    function __len__.

    :param target_cls: the class where the full parameter space is to be built.

    .. note::
        The __init__ method is aware of the implementation details of the
        regression test metaclass. This is required to retrieve the parameter
        spaces from the base classes, and also the local parameter space from
        the target class.
    '''

    def __init__(self, target_cls=None):
        self._params = {}

        # If a target class is provided, build the param space for it
        if target_cls:
            assert hasattr(target_cls, '_rfm_local_param_space')
            assert isinstance(target_cls._rfm_local_param_space,
                              LocalParamSpace)

            # Inherit the parameter spaces from the direct parent classes
            for base in filter(lambda x: hasattr(x, 'param_space'),
                               target_cls.__bases__):
                self.join(base._rfm_param_space)

            # Extend the parameter space with the local parameter space
            for name, p in target_cls._rfm_local_param_space.items():
                self._params[name] = (
                    p.filter_params(self._params.get(name, ())) + p.values
                )

            # Make sure that none of the parameters clashes with the target
            # class namespace
            target_namespace = set(dir(target_cls))
            for key in self._params:
                if key in target_namespace:
                    raise ReframeSyntaxError(
                        f'parameter {key!r} clashes with other variables'
                        f' present in the namespace from class '
                        f'{target_cls.__qualname__!r}'
                    )

        # Internal parameter space usage tracker
        self.__unique_iter = iter(self)

    def join(self, other):
        '''Join two parameter spaces into one

        Join two different parameter spaces into a single one. Both parameter
        spaces must be an instance ot the ParamSpace class. This method will
        raise an error if a parameter is defined in the two parameter spaces
        to be merged.

        :param other: instance of the ParamSpace class
        '''
        for key in other.params:
            # With multiple inheritance, a single parameter
            # could be doubly defined and lead to repeated
            # values
            if (key in self._params and
                self._params[key] != () and
                other.params[key] != ()):

                raise ReframeSyntaxError(f'parameter space conflict: '
                                         f'parameter {key!r} already defined '
                                         f'in {b.__qualname__!r}')

            self._params[key] = (
                other.params.get(key, ()) + self._params.get(key, ())
            )

    def __iter__(self):
        '''Create a generator object to iterate over the parameter space

        :return: generator object to iterate over the parameter space.
        '''
        yield from itertools.product(*(p for p in self._params.values()))

    @property
    def params(self):
        return self._params

    @property
    def unique_iter(self):
        '''Expose the internal iterator as read-only'''
        return self.__unique_iter

    def __len__(self):
        '''Returns the number of all possible parameter combinations.

        Method to calculate the test's parameter space length (i.e. the number
        of all possible parameter combinations). If the RegressionTest
        has no parameters, the length is 1.

        .. note::
           If the test is an abstract test (i.e. has undefined parameters in
           the parameter space), the returned parameter space length is 0.

        :return: length of the parameter space
        '''
        if not self._params:
            return 1

        return functools.reduce(
            lambda x, y: x*y,
            (len(p) for p in self._params.values())
        )

    def __getitem__(self, key):
        return self._params.get(key, ())

    def is_empty(self):
        return self._params == {}
