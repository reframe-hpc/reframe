# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class Namd_BaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for the NAMD Test.

    NAMD is a parallel molecular dynamics code designed for
    high-performance simulation of large biomolecular systems.
    Based on Charm++ parallel objects, NAMD scales to hundreds of
    cores for typical simulations and beyond 500,000 cores for the
    largest simulations. NAMD uses the popular molecular graphics
    program VMD for simulation setup and trajectory analysis,
    but is also file-compatible with AMBER, CHARMM, and X-PLOR.
    (see ks.uiuc.edu/Research/namd/)

    The presented abstract run-only class checks the NAMD perfomance.
    To do this, it is necessary to define in tests  the reference values
    of energy and possible deviations from this value. This data is used
    to check if the task is being executed correctly, that is, the final
    energy is correct (approximately the reference). The default
    assumption is that NAMD is already installed on the device under test.
    '''

    #: Reference value of energy, that is used for the comparison
    #: with the execution ouput on the sanity step. The absolute
    #: difference between final energy value and reference value
    #: should be smaller than energy_tolerance
    #:
    #: :type: str
    #: :default: :class:`required`
    energy_value = variable(float)

    #: Maximum deviation from the reference value of energy,
    #: that is acceptable.
    #:
    #: :default: :class:`required`
    energy_tolerance = variable(float)

    #: :default: :class:`required`
    executable = required

    executable = 'namd2'
    energy_value = -2451359.5
    energy_tolerance = 2720.

    @run_after('init')
    def source_install(self):
        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'NAMD', 'prod')

    @run_before('performance')
    def set_the_performance_dict(self):
        self.perf_variables = {self.scale:
                               sn.make_performance_function(
                                   sn.avg(
                                       sn.extractall(
                                           r'Info: Benchmark time: \S+ CPUs'
                                           r' \S+ s/step (?P<days_ns>\S+) '
                                           r'days/ns \S+ MB memory',
                                           self.stdout, 'days_ns', float)),
                                   'days/ns')}

    @performance_function('days/ns', perf_key='perf')
    def set_perf_patterns(self):
        return sn.avg(sn.extractall(
                          r'Info: Benchmark time: \S+ CPUs \S+ '
                          r's/step (?P<days_ns>\S+) days/ns \S+ MB memory',
                          self.stdout, 'days_ns', float))

    @sanity_function
    def assert_energy_readout(self):
        '''Assert the obtained energy meets the specified tolerances.'''

        energy = sn.avg(sn.extractall(
            r'ENERGY:([ \t]+\S+){10}[ \t]+(?P<energy>\S+)',
            self.stdout, 'energy', float)
        )
        energy_diff = sn.abs(energy - self.energy_value)
        ref_ener_diff = sn.abs(self.energy_tolerance)
        return sn.all([
            sn.assert_eq(sn.count(sn.extractall(
                         r'TIMING: (?P<step_num>\S+)  CPU:',
                         self.stdout, 'step_num')), 50),
            sn.assert_lt(energy_diff, self.energy_tolerance)
        ])
