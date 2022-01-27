# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to build extensible parameterized tests.
#

import copy
import itertools

import reframe.core.namespaces as namespaces
import reframe.utility as utils
from reframe.core.exceptions import ReframeSyntaxError


class TestParam:
    '''Regression test paramter class.

    Stores the attributes of a regression test parameter as defined directly
    in the test definition. These attributes are the parameter's name,
    values, and inheritance behaviour. This class should be thought of as a
    temporary storage for these parameter attributes, before the full final
    parameter space is built.

    :meta private:
    '''

    def __init__(self, values=None,
                 inherit_params=False, filter_params=None, fmt=None):
        if values is None:
            values = []

        if not inherit_params:
            # By default, filter out all the parameter values defined in the
            # base classes.
            def filter_params(x):
                return ()
        elif filter_params is None:
            # If inherit_params==True, inherit all the parameter values from
            # the base classes as default behaviour.
            def filter_params(x):
                return x

        self.values = tuple(values)

        # Validate and set the filter_params function
        if (not callable(filter_params) or
            not utils.is_trivially_callable(filter_params, non_def_args=1)):
            raise TypeError("'filter_params' argument must be a callable "
                            "accepting a single argument")

        self.filter_params = filter_params

        # Validate and set the alternative function
        if fmt is None:
            def fmt(x):
                return x

        if (not callable(fmt) or
            not utils.is_trivially_callable(fmt, non_def_args=1)):
            raise TypeError("'fmt' argument must be a callable "
                            "accepting a single argument")

        self.__fmt_fn = fmt

    @property
    def format(self):
        return self.__fmt_fn

    def update(self, other):
        '''Update this parameter from another one.

        The values from the other parameter will be filtered according to the
        filter function of this one and prepended to this parameter's values.
        '''

        try:
            filt_vals = self.filter_params(other.values)
        except Exception:
            raise
        else:
            try:
                self.values = tuple(filt_vals) + self.values
            except TypeError:
                raise ReframeSyntaxError(
                    f"'filter_param' must return an iterable"
                ) from None

    def is_abstract(self):
        return len(self.values) == 0


class ParamSpace(namespaces.Namespace):
    '''Regression test parameter space

    The parameter space is stored as a dictionary (self.params), where the
    keys are the parameter names and the values are tuples with all the
    available values for each parameter. The __init__ method in this class
    takes the optional argument ``target_cls``, which is the regression test
    class that the parameter space is being built for. If no target class is
    provided, the parameter space is initialized as empty.

    All the parameter combinations are stored under ``__param_combinations``.
    This enables random-access to any of the available parameter combinations
    through the ``__getitem__`` method.
    '''

    def __init__(self, target_cls=None, illegal_names=None):
        super().__init__(target_cls, illegal_names,
                         ns_name='_rfm_param_space',
                         ns_local_name='_rfm_local_param_space')

        # Store all param combinations to allow random access.
        self.__param_combinations = tuple(
            itertools.product(
                *(copy.deepcopy(p.values) for p in self.params.values())
            )
        )

        # Map the parameter names to the position they are stored in the
        # parameter space
        self._position = {name: idx for idx, name in enumerate(self.params)}

    def join(self, other, cls):
        '''Join other parameter space into the current one.

        Join two different parameter spaces into a single one. Both parameter
        spaces must be an instance ot the ParamSpace class. This method will
        raise an error if a parameter is defined in the two parameter spaces
        to be merged.

        :param other: instance of the ParamSpace class.
        :param cls: the target class.
        '''
        for name in other.params:
            # With multiple inheritance, a single parameter
            # could be doubly defined and lead to repeated
            # values
            if self.defines(name) and other.defines(name):
                raise ReframeSyntaxError(
                    f'parameter space conflict: '
                    f'parameter {name!r} is defined in more than '
                    f'one base class of class {cls.__qualname__!r}'
                )

            if not self.defines(name):
                # If we do not define the parameter, take it from other
                self.params[name] = other.params[name]

    def extend(self, cls):
        '''Extend the parameter space with the local parameter space.'''

        local_param_space = getattr(cls, self.local_namespace_name, dict())
        for name, p in local_param_space.items():
            if name in self.params:
                p.update(self.params[name])

            self.params[name] = p

        # Clear the local param space
        local_param_space.clear()

        # If any previously declared parameter was defined in the class body
        # by directly assigning it a value, raise an error. Parameters must be
        # changed using the `x = parameter([...])` syntax.
        for key, values in cls.__dict__.items():
            if key in self.params:
                raise ReframeSyntaxError(
                    f'parameter {key!r} must be modified through the built-in '
                    f'parameter type'
                )

    def inject(self, obj, cls=None, params_index=None):
        '''Insert the params in the regression test.

        Create and initialize the regression test parameters as object
        attributes. The values assigned to these parameters exclusively depend
        on the value of params_index. This argument is simply an index to a
        a given parametere combination. If params_index is left with its
        default value, the regression test parameters are initialized as
        None.

        :param obj: The test object.
        :param cls: The test class.
        :param params_index: index to a point in the parameter space.
        '''
        # Set the values of the test parameters (if any)
        if self.params and params_index is not None:
            try:
                # Get the parameter values for the specified variant
                param_values = self.__param_combinations[params_index]
            except IndexError as no_params:
                raise RuntimeError(
                    f'parameter space index out of range for '
                    f'{obj.__class__.__qualname__}'
                ) from None
            else:
                for index, key in enumerate(self.params):
                    setattr(obj, key, param_values[index])

        else:
            # Otherwise init the params as None
            for key in self.params:
                setattr(obj, key, None)

    @property
    def params(self):
        return self._namespace

    def defines(self, name):
        '''Return True if parameter is defined.

        A parameter is defined if it exists in the namespace and it is not
        abstract.
        '''
        return name in self.params and not self.params[name].is_abstract()

    def __iter__(self):
        '''Create a generator object to iterate over the parameter space

        The parameters must be deep-copied to prevent an instance from
        modifying the class parameter space.

        :return: generator object to iterate over the parameter space.
        '''
        yield from self.__param_combinations

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
        if not self.params:
            return 1

        return len(self.__param_combinations)

    def __getitem__(self, key):
        '''Access an element in the parameter space.

        If the key is an integer, this will be interpreted as a point in the
        parameter space and this function will return a mapping of the
        parameter names and their corresponding values. If the key is a
        parameter name, it will instead return all the values assigned to that
        parameter.

        If the key is an integer, this function will raise an
        :class:`IndexError` if the key is out of bounds.

        '''
        if isinstance(key, int):
            ret = {}
            val = self.__param_combinations[key]
            for i, name in enumerate(self.params):
                ret[name] = val[i]

            return ret

        try:
            return self.params[key].values
        except KeyError:
            return ()

    def is_empty(self):
        return self.params == {}

    def get_variant_nums(self, **conditions):
        '''Filter the paramter indices with a given set of conditions.

        The conditions are passed as key-value pairs, where the keys are the
        parameter names to apply the filtering on and the values are functions
        that expect the parameter's value as the sole argument.

        :returns: the indices of the matching parameters in the parameter
            space.

        '''
        candidates = range(len(self))
        if not conditions:
            return list(candidates)

        # Validate conditions
        for param, cond in conditions.items():
            if param not in self:
                raise NameError(
                    f'no such parameter: {param!r}'
                )
            elif not callable(cond):
                # Convert it to the identity function
                val = cond

                def cond(x):
                    return x == val
            elif not utils.is_trivially_callable(cond, non_def_args=1):
                raise ValueError(
                    f'condition on {param!r} must be a callable accepting a '
                    f'single argument'
                )

            def param_val(variant):
                return self._get_param_value(param, variant)

            # Filter the given condition
            candidates = [v for v in candidates if cond(param_val(v))]

        return candidates

    def _get_param_value(self, name, variant):
        '''Get the a parameter's value for a given variant.

        In this context, a variant is a point in the parameter space.
        The name argument is simply the parameter name
        '''
        return self.__param_combinations[variant][self._position[name]]
