# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class Gromacs_BaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for the Gromacs Test.

    GROMACS is a versatile package to perform molecular dynamics,
    i.e. simulate the Newtonian equations of motion for systems
    with hundreds to millions of particles.

    It is primarily designed for biochemical molecules like proteins,
    lipids and nucleic acids that have a lot of complicated bonded
    interactions, but since GROMACS is extremely fast at calculating
    the nonbonded interactions (that usually dominate simulations)
    many groups are also using it for research on non-biological
    systems, e.g. polymers (see gromacs.org).

    The presented abstract run-only class checks the Gromacs perfomance.
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

    #: Amber output file.
    #:
    #: :default: : 'amber.out'
    output_file = variable(str, value='md.log')

    executable = 'gmx_mpi'
    energy_value = -3270799.9
    energy_tolerance = 0.001

    @run_after('init')
    def source_install(self):
        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Gromacs', 'herflat')

    @run_after('init')
    def set_keep_files(self):
        self.keep_files = [self.output_file]

    @performance_function('ns/day', perf_key='perf')
    def set_perf_patterns(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                self.output_file, 'perf', float)

    @run_before('performance')
    def set_the_performance_dict(self):
        self.perf_variables = {self.mode:
                               sn.make_performance_function(
                                   sn.extractsingle(
                                       r'Performance:\s+(?P<perf>\S+)',
                                       self.output_file, 'perf', float),
                                   'ns/day')}

    @sanity_function
    def assert_energy_readout(self):
        '''Assert the obtained energy meets the specified tolerances.'''
        energy = sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                  r'\s+Conserved En\.\s+Temperature\n'
                                  r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                  r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                  self.output_file, 'energy', float, item=-1)
        ref_ener_diff = sn.abs(self.energy_value *
                               self.energy_tolerance)

        return sn.all([
            sn.assert_found('Finished mdrun', self.output_file),
            sn.assert_lt(self.energy_tolerance, ref_ener_diff)
        ])
