# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible parameterized tests.
#

from reframe.core.exceptions import ReframeSyntaxError


class _TestParameter:
    '''Regression test paramter class.

    Stores the attributes of a regression test parameter as defined directly
    in the test definition. These attributes are the parameter's name,
    values, and inheritance behaviour. This class should be thought of as a
    temporary storage for these parameter attributes, before the full final
    parameter space is built.

    :param name: parameter name
    :param values: parameter values. If no values are passed, the parameter is
        considered as declared but not defined (i.e. an abstract parameter).
    :param inherit_params: If false, this parameter is marked to not inherit
        any values for the same parameter that might have been defined in a
        parent class.
    :param filter_params: Function to filter/modify the inherited parameter
        values from the parent classes. This only has an effect if used with
        inherit_params=True.
    '''

    def __init__(self, name, *values,
                 inherit_params=False, filter_params=None):
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
        self.values = values
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
                f'parameter {name} already defined in this class'
            )

    def add_param(self, name, *values, **kwargs):
        '''Insert a new regression test parameter in the local parameter space.

        If the parameter is already present in the dictionary, raise an error.
        See the _TestParameter class for further information on the
        function arguments.
        '''
        self[name] = _TestParameter(name, *values, **kwargs)

    @property
    def params(self):
        return self._params


def _merge_parameter_spaces(bases):
    '''Merge the parameter space from multiple classes.

    Joins the parameter space of multiple classes into a single parameter
    space. This method allows multiple inheritance, as long as a parameter is
    not doubly defined in two or more different parameter spaces.

    :param bases: iterable containing the classes from which to merge the
        parameter space.

    :returns: merged parameter space.
    '''
    # Temporary dict where we build the parameter space from the base
    # classes
    param_space = {}

    # Iterate over the base classes and inherit the parameter space
    for b in bases:
        base_params = getattr(b, '_rfm_params', ())
        for key in base_params:
            # With multiple inheritance, a single parameter
            # could be doubly defined and lead to repeated
            # values.
            if (key in param_space and (
                param_space[key] != () and base_params[key] != ()
               )):

                raise ReframeSyntaxError(f'parameter space conflict: '
                                         f' parameter {key!r} already defined '
                                         f'in {b.__qualname__!r}')

            param_space[key] = (
                base_params.get(key, ()) + param_space.get(key, ())
            )

    return param_space


def _extend_parameter_space(param_space, local_param_space):
    '''Extend a given parameter space with a local parameter space.

    Each parameter is dealt with independently, given that each parameter
    has its own inheritance behaviour defined in the local parameter space
    (see the
    :class:`reframe.core.parameters_TestParameter` class).

    :param param_space: an existing parameter space. This **must** have been
        generated with
        :meth:`reframe.core.parameters._merge_parameter_spaces`.
    :param local_param_space: a local parameter space from a regression test.
        This must be an instance of the class
        :class:`reframe.core.parameters.LocalParamSpace`.
    '''
    # The argument local_param_space must be an instance of LocalParamSpace
    assert isinstance(local_param_space, LocalParamSpace)

    # Loop over the local parameter space.
    for name, p in local_param_space.params.items():
        param_space[name] = (
            p.filter_params(param_space.get(name, ())) + p.values
        )

    return param_space


def build_parameter_space(cls):
    ''' Builder of the full parameter space of a regression test.

    Handles the full parameter space build, inheriting the parameter spaces
    form the base classes, and extending these with the local parameter space
    of the class (stored in cls._rfm_local_param_space). This method is called
    during the class object initialization (i.e. the __init__ method of the
    regression test metaclass). This method has three main steps, which are
    (in order of execution) the inheritance of the parameter spaces from the
    base classes, the extension of the inherited parameter space with the
    local parameter space, and lastly, a check to ensure that none of the
    parameter names clashes with any of the class attributes existing in the
    regression test class.

    :param cls: the class where the full parameter space is to be built.

    :returns: dictionary containing the full parameter space. The keys are the
        parameter names and the values are tuples containing all the values
        for each of the parameters.
    '''
    param_space = _extend_parameter_space(
        _merge_parameter_spaces(cls.__bases__), cls._rfm_local_param_space
    )

    trgt_namespace = set(dir(cls))
    for key in param_space:
        if key in trgt_namespace:
            raise ReframeSyntaxError(f'parameter {key!r} clashes with other '
                                     f'variables present in the namespace '
                                     f'from class {cls.__qualname__!r}')

    setattr(cls, '_rfm_params', param_space)
