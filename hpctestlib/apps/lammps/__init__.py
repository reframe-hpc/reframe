# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["LammpsBaseCheck"]


class LAMMPSBaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the LAMMPS Test. It is adapted to check the
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

    @sanity_function
    def set_sanity_patterns(self):
        '''Standart sanity check for the LAMMPS. Compare the
        reference value of energy with obtained from the executed
        program.
        '''

        energy = sn.extractsingle(
            r'\s+500000(\s+\S+){3}\s+(?P<energy>\S+)\s+\S+\s\n',
            self.stdout, 'energy', float)
        energy_diff = sn.abs(energy - self.reference_value)
        ref_ener_diff = sn.abs(self.reference_value *
                               self.reference_difference)

        return sn.all([
            sn.assert_found(r'Total wall time:', self.stdout),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
