# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# attribute extension to the RegressionTest classes.
#


class RegressionTestParameter:
    '''
    Regression test paramter class.
    Stores the attributes of a regression test parameter needed to build
    the full parameter space of a regression test. These are the parameter's
    name, values, and inheritance behaviour.

    This is used by the ParameterStagingArea class below.
    '''

    def __init__(self, name, *values,
                 inherit_params=False, filter_params=None):
        '''
        name: parameter name
        values: parameter values. If no values are passed, the parameter is
            considered as declared but not defined (i.e. an abstract param).
        inherit_params: If false, this parameter is marked to not inherit any
            values that might have been defined in a parent class.
        filter_params: Function to filter/modify the inherited parameter values
             from the parent classes. This only has an effect if used with
             inherit_params=True.
        '''
        # By default, do not filter any of the inherited parameter values.
        if filter_params is None:
            def filter_params(x):
                return x

        self.name = name
        self.values = values
        self.inherit_params = inherit_params
        self.filt_params = filter_params


class ParameterStagingArea:
    '''
    Staging area of the regression test parameters used to build the
    regression test parameter space. This staging area is simply a set of
    regression test parameters declared in a given class derived from the
    RegressionTest class. This set does not include any other parameters
    that might have been declared/defined in any of the parent classes. The
    parameter staging area must not be confused with the regression test's
    parameter space.

    Example: In the pseudo-code below, the parameter staging area of A is {P0},
    and the staging area of B is {P1}. However, the parameter space of A is
    still {P0}, and the parameter space of B is {P0, P1}.

        class A(RegressionTest):
            -> define parameter P0 with value X.

        class B(A):
            -> define parameter P1 with value Y.


    The parameter staging area is populated during the class body execution.
    This is done using class directives that call the member function
    add_regression_test_parameter.

    The regression test's parameter space is assembled during the class object
    initialization (i.e. the __init__ method of the metaclass), where a call
    to the member function build_parameter_space is made. This member function
    handles the parameter inheritance from the parent classes.
    '''

    def __init__(self):
        self.staging_area = {}

    def add_regression_test_parameter(self, name, *values, **kwargs):
        '''
        Insert a new regression test parameter in the staging area.
        If the parameter is already present in the dictionary, raise an error.
        See the RegressionTestParameter class for further information on the
        function arguments.
        '''
        if name not in self.staging_area:
            self.staging_area[name] = RegressionTestParameter(
                name, *values, **kwargs
            )
        else:
            raise ValueError(
                'Cannot double-define a parameter in the same class.'
            ) from None

    def _inherit_parameter_space(self, bases, param_space_key):
        '''
        Inherit the parameter space from the base clases. Note that the
        parameter space is simply a dictionary where the keys are the
        parameter names and the values of this dictionary are tuples with the
        values of the associated parameter.

        bases: iterable containing the parent classes.
        param_space_key: class attribute name under which the parameter space
            is stored.
        '''
        # Temporary dict where we build the parameter space from the base
        # clases
        temp_parameter_space = {}

        # Iterate over the base classes and inherit the parameter space
        for b in bases:
            base_params = b.__dict__.get(param_space_key, ())
            for key in base_params:
                # With multiple inheritance, a single parameter
                # could be doubly defined and lead to repeated
                # values.
                if key in temp_parameter_space:
                    if not (temp_parameter_space[key] == () or
                            base_params[key] == ()):
                        raise KeyError(f'Parameter space conflict '
                                       f'(on {key}) due to '
                                       f'multiple inheritance.'
                                       ) from None

                temp_parameter_space[key] = base_params.get(
                    key, ()) + temp_parameter_space.get(key, ())

        return temp_parameter_space

    def _extend_parameter_space(self, parameter_space):
        '''
        Add the parameters from the staging area into the parameter space
        inherited from the parent classes.

        Each parameter is dealt with independently, following the inheritance
        behaviour set in the RegressionTestParameter class (i.e. inherit+filter
        operations as defined for each parameter).
        '''
        # Loop over the staging area.
        for name, p in self.staging_area.items():
            parameter_space[name] = p.filt_params(
                parameter_space.get(name, ())) + p.values if (
                    p.inherit_params) else p.values

    def build_parameter_space(self, target_cls, bases, param_space_key):
        '''
        Compiles the full test parameter space by joining the parameter spaces
        from the base clases, and extending that with the parameters present
        in the staging area.

        target_cls: class to test if the parameter space overlaps with its own
            namespace.
        bases: iterable containing the parent classes.
        param_space_key: class attribute name under which the parameter space
            is stored.
        '''
        # Inherit from the bases
        param_space = self._inherit_parameter_space(bases, param_space_key)

        # Extend with what was added in the current class
        self._extend_parameter_space(param_space)

        trgt_namespace = set(dir(target_cls))
        for key in param_space:
            if key in trgt_namespace:
                raise AttributeError(f'Attribute {key} clashes with other '
                                     f'variables present in the namespace '
                                     f'from class {target_cls.__qualname__}')

        return param_space
