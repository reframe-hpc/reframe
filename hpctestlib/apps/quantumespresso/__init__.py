# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["QuantumESPRESSOBaseCheck"]


class QuantumESPRESSOBaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for the Quantum ESPRESSO Test. It is adapted to check the
    correctness of the execution of a given script (in terms of
    energy received).
    '''

    # Name of executed script. Required variable
    input_file = variable(str)

    # Required variable
    num_tasks_per_node = required

    # Reference value of energy, that is used for the comparison.
    # Required variable
    reference_value = variable(float)

    # Maximum deviation from the reference  value of energy,
    # that is acceptable. Required variable
    reference_difference = variable(float)

    @run_after('setup')
    def set_executable_opts(self):
        '''Set the executable options for the Quantum ESPRESSO. Determine the
        using of input file.
        '''
        self.executable_opts = ['-in', self.input_file]


    @sanity_function
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
        ])

        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)

        energy_diff = sn.abs(energy-self.reference_value)
        ref_ener_diff = sn.abs(self.reference_difference)
        return sn.all([
            self.sanity_patterns,
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
