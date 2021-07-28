# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["VaspBaseCheck"]


class VASPBaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for the VASP Test. It is adapted to check the
    correctness of the execution of a given script (in terms of
    force received).
    '''

    # Required variables
    num_tasks_per_node = required

    # Reference value of force, that is used for the comparison.
    # Required variable
    reference_value = variable(float)

    # Maximum deviation from the reference  value of force,
    # that is acceptable. Required variable
    reference_difference = variable(float)

    # Name of the keep files for the case of VASP is standart
    keep_files = ['OUTCAR']

    @sanity_function
    def set_sanity_patterns(self):
        '''Standart sanity check for the VASP. Compare the
        reference value of force with obtained from the executed
        program.
        '''

        force = sn.extractsingle(r'1 F=\s+(?P<result>\S+)',
                                 self.stdout, 'result', float)
        force_diff = sn.abs(force - self.reference_value)
        ref_force_diff = sn.abs(self.reference_value *
                                self.reference_difference)

        return sn.assert_lt(force_diff, ref_force_diff)
