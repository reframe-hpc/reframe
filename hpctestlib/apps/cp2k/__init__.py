# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["Cp2k"]


class Cp2k(rfm.RunOnlyRegressionTest):
    '''Base class for the CP2K Test.

    CP2K is a quantum chemistry and solid state physics software
    package that can perform atomistic simulations of solid state,
    liquid, molecular, periodic, material, crystal, and biological
    systems. CP2K provides a general framework for different modeling
    methods such as DFT using the mixed Gaussian and plane waves
    approaches GPW and GAPW.  (see cp2k.org).

    The presented abstract run-only class checks the work of cp2k.
    To do this, it is necessary to define in tests  the reference
    values of energy and possible deviations from this value.
    This data is used to check if the task is being executed
    correctly, that is, the final energy is correct
    (approximately the reference). The default assumption
    is that CP2K is already installed on the device under test.
    '''

    #: Reference value of energy, that is used for the comparison
    #: with the execution ouput on the sanity step. Final value of
    #: energy should be approximately the same
    #:
    #: :default: :class:`required`
    energy_value = variable(float)

    #: Maximum deviation from the reference value of energy,
    #: that is acceptable.
    #:
    #: :default: :class:`required`
    energy_tolerance = variable(float)

    # Required variable
    num_tasks_per_node = required

    @sanity_function
    def set_sanity_patterns(self):
        '''Assert the obtained energy meets the specified tolerances.'''

        energy = sn.extractsingle(
            r'\s+ENERGY\| Total FORCE_EVAL \( QS \) '
            r'energy [\[\(]a\.u\.[\]\)]:\s+(?P<energy>\S+)',
            self.stdout, 'energy', float, item=-1
        )
        energy_diff = sn.abs(energy - self.energy_value)
        ref_ener_diff = sn.abs(self.energy_tolerance)

        return sn.all([
            sn.assert_found(r'PROGRAM STOPPED IN', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?i)(?P<step_count>STEP NUMBER)',
                self.stdout, 'step_count')), 10),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
