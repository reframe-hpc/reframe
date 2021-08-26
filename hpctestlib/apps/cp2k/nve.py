# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class Cp2k_NVE(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the CP2K NVE Test.

    CP2K is a quantum chemistry and solid state physics software
    package that can perform atomistic simulations of solid state,
    liquid, molecular, periodic, material, crystal, and biological
    systems. CP2K provides a general framework for different modeling
    methods such as DFT using the mixed Gaussian and plane waves
    approaches GPW and GAPW.  (see cp2k.org).

    The presented abstract run-only class checks the perfomance of cp2k.
    To do this, it is necessary to define in tests  the reference
    values of energy and possible deviations from this value.
    This data is used to check if the task is being executed
    correctly, that is, the final energy is correct
    (approximately the reference). The default assumption
    is that CP2K is already installed on the device under test.
    '''

    # Parameter pack containing the platform ID
    platform_name = parameter(['cpu', 'gpu'])

    #: Reference value of energy, that is used for the comparison
    #: with the execution ouput on the sanity step. The absolute
    #: difference between final energy value and reference value
    #: should be smaller than energy_tolerance
    #:
    #: :type: str
    #: :default: :class:`required`
    energy_value = -4404.2323

    #: Maximum deviation from the reference value of energy,
    #: that is acceptable.
    #:
    #: :default: :class:`required`
    energy_tolerance = 1E-04

    #: :default: :class:`required`
    num_tasks_per_node = required

    #: :default: :class:`required`
    executable = required

    executable = 'cp2k.psmp'
    executable_opts = ['H2O-256.inp']

    @performance_function('s', perf_key='time')
    def set_perf_patterns(self):
        return sn.extractsingle(r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                                self.stdout, 'perf', float)

    @run_before('performance')
    def set_the_performance_dict(self):
        self.perf_variables = {self.mode:
                               sn.make_performance_function(
                                   sn.extractsingle(
                                       r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>' +
                                       r'\S+)', self.stdout, 'perf',
                                       float), 's')}

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
