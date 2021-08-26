# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class LAMMPS_NVE(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the LAMMPS NVE Test.

    LAMMPS is a classical molecular dynamics code with a focus
    on materials modeling. It's an acronym for Large-scale
    Atomic/Molecular Massively Parallel Simulator.

    LAMMPS has potentials for solid-state materials (metals,
    semiconductors) and soft matter (biomolecules, polymers)
    and coarse-grained or mesoscopic systems. It can be used
    to model atoms or, more generically, as a parallel particle
    simulator at the atomic, meso, or continuum scale.
    (see lammps.org)

    The presented abstract run-only class checks the LAMMPS perfomance.
    To do this, it is necessary to define in tests the name
    of the running script (input file), as well as set the
    reference values of energy and possible deviations from this
    value. This data is used to check if the task is being
    executed correctly, that is, the final energy is correct
    (approximately the reference). The default assumption is that
    LAMMPS is already installed on the device under test.
    '''

    #: Name of executed script
    #:
    #: :default: :class:`required`
    input_file = variable(str)

    #: Reference value of energy, that is used for the comparison
    #: with the execution ouput on the sanity step. The absolute
    #: difference between final energy value and reference value
    #: should be smaller than energy_tolerance
    #:
    #: :type: str
    #: :default: :class:`required`
    energy_value = -4.6195

    #: Maximum deviation from the reference value of energy,
    #: that is acceptable.
    #:
    #: :default: :class:`required`
    energy_tolerance = 6.0E-04

    #: :default: :class:`required`
    num_tasks_per_node = required

    #: Parameter pack containing the platform ID and input file
    platform = parameter([
        ('cpu', 'in.lj.cpu'),
        ('gpu', 'in.lj.gpu')
    ])

    @run_after('init')
    def unpack_platform_parameter(self):
        '''Set the executable and input file.'''

        self.platform_name, self.input_file = self.platform

    @run_after('init')
    def source_install(self):
        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'LAMMPS')

    @performance_function('timesteps/s', perf_key = 'nve')
    def set_perf_patterns(self):
        return  sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                 self.stdout, 'perf', float)

    @run_before('performance')
    def set_the_performance_dict(self):
        self.perf_variables = {self.mode:
                                sn.make_performance_function(
                                sn.extractsingle(
                                   r'\s+(?P<perf>\S+) timesteps/s',
                                   self.stdout, 'perf', float), 'timesteps/s')}

    @sanity_function
    def set_sanity_patterns(self):
        '''Standart sanity check for the LAMMPS. Compare the
        reference value of energy with obtained from the executed
        program.
        '''

        energy = sn.extractsingle(
            r'\s+500000(\s+\S+){3}\s+(?P<energy>\S+)\s+\S+\s\n',
            self.stdout, 'energy', float)
        energy_diff = sn.abs(energy - self.energy_value)
        ref_ener_diff = sn.abs(self.energy_value *
                               self.energy_tolerance)

        return sn.all([
            sn.assert_found(r'Total wall time:', self.stdout),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
