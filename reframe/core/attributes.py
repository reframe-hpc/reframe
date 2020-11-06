# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# attribute extension to the RegressionTest classes.
#


class InputParameter:
    '''
    Input paramter and (name + values) and other attributes to deal with the
    test parameter inheritance/chaining.
    '''

    def __init__(self, name, values=None, inherit_params=False, filt_params=None):
        '''
        name: parameter name
        values: parameter values
        inherit_params: If false, it overrides all previous values set for the parameter.
        filt_params: Function to filter/modify the inherited values for the parameter.
             It only has an effect if used with inherit_params=True.
        '''
        # If the values are None (or an empty list), the parameter is considered as
        # declared but not defined (i.e. an abstract parameter).
        if values is None:
            values = []

        # The values arg must be a list.
        if not isinstance(values, list):
            raise ValueError(
                f'Parameter values must be defined in a list.') from None

        # Default filter is no filter.
        if filt_params is None:
            def filt_params(x): return x

        self.name = name
        self.values = values
        self.inherit_params = inherit_params
        self.filt_params = filt_params

    def is_undef(self):
        '''
        Handy function to check if the parameter is undefined (empty).
        '''
        if self.values == []:
            return True

        return False


class ParameterPack:
    '''
    Bundle containing the test parameters inserted in each test (class).
    The parameters are store in a dictionary for a more efficient lookup.
    The add method is the interface used to add new parameters to the reframe test.
    '''

    def __init__(self):
        self.parameter_map = {}
        self.purge_parameter_space = False
        self.parameter_blacklist = set()

    def add(self, name, values=None, inherit_params=False, filt_params=None):
        '''
        Insert a new parameter in the dictionary.
        If the parameter is already present in it, raise an error.
        '''
        if name not in self.parameter_map:
            self.parameter_map[name] = InputParameter(
                name, values, inherit_params, filt_params)
        else:
            raise ValueError(
                'Cannot double-define a parameter in the same class.')

    def purge_all_parameters(self):
        '''
        Set the purge_parameter_space flag to true, which blocks the inheritance
        of the full parameter space.
        '''
        self.purge_parameter_space = True

    def purge_parameters(self, *params_to_purge):
        '''
        Override the inheritance of a given set of parameters.
        '''
        for param in params_to_purge:
            self.parameter_blacklist.add(param)


class RegressionTestAttributes:
    '''
    Storage class hosting all the reframe class attributes.
    '''

    def __init__(self):
        self._rfm_parameter_stage = ParameterPack()

    def get_parameter_stage(self):
        return self._rfm_parameter_stage.parameter_map

    def _purge_all_parameters(self):
        return self._rfm_parameter_stage.purge_parameter_space

    def _parameter_blacklist(self):
        return self._rfm_parameter_stage.parameter_blacklist

    def _inherit_parameter_space(self, bases):
        '''
        Build the parameter space from the base clases.
        '''
        # Temporary dict where we build the parameter space from the base clases
        temp_parameter_space = {}

        # Iterate over the base classes and inherit the parameter space
        if not self._purge_all_parameters():
            for b in bases:
                if hasattr(b, '_rfm_params'):
                    base_params = b._rfm_params
                    for key in base_params:
                        if ((key in self.get_parameter_stage() and
                             not self.get_parameter_stage().get(key).inherit_params) or
                            key in self._parameter_blacklist()):

                            # Do not inherit a given parameter if the current class wants to override it.
                            pass

                        else:

                            # With multiple inheritance, a single parameter could be doubly defined
                            # and lead to repeated values.
                            if key in temp_parameter_space:
                                if not (temp_parameter_space[key] == [] or
                                        base_params[key] == []):
                                    raise KeyError(f'Parameter space conflict (on {key}) '
                                                   f'due to multiple inheritance.') from None

                            temp_parameter_space[key] = base_params.get(
                                key, []) + temp_parameter_space.get(key, [])

                else:
                    # The base class does not have the attribute cls._rfm_params
                    pass

        return temp_parameter_space

    def _extend_parameter_space(self, parameter_space):
        '''
        Add the parameters from the parameter stage into the existing parameter space.
        Do the inherit+filter operations as defined in the for each input parameter in the
        parameter stage.
        '''
        # Loop over the parameter stage. Each element is an instance of InputParameter.
        for name, p in self.get_parameter_stage().items():
            parameter_space[name] = p.filt_params(parameter_space.get(name, [])) + p.values if (
                p.inherit_params) else p.values

    def build_parameter_space(self, bases):
        '''
        Compiles the full test parameter space by joining the parameter spaces from the base clases,
        and extending that with the parameters present in the parameter stage.
        '''
        # Inherit from the bases
        param_space = self._inherit_parameter_space(bases)

        # Extend with what was added in the current class
        self._extend_parameter_space(param_space)

        return param_space

    @staticmethod
    def namespace_clash_check(dict_a, dict_b, name=None):
        '''
         Check that these two dictionaries do not have any overlapping keys.
        '''
        if len(dict_a) > len(dict_b):
            long_dict = dict_a
            short_dict = dict_b
        else:
            long_dict = dict_b
            short_dict = dict_a

        name = '__unknown__' if name is None else name
        for key in short_dict:
            if key in long_dict:
                raise AttributeError(f'Attribute {key} clashes with other variables'
                                     f'present in the namespace of class {name}')
