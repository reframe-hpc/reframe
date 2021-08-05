# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["LAMMPS"]


class LAMMPS(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the LAMMPS Test.

    LAMMPS is a classical molecular dynamics code with a focus
    on materials modeling. It's an acronym for Large-scale
    Atomic/Molecular Massively Parallel Simulator.

    LAMMPS has potentials for solid-state materials (metals,
    semiconductors) and soft matter (biomolecules, polymers)
    and coarse-grained or mesoscopic systems. It can be used
    to model atoms or, more generically, as a parallel particle
    simulator at the atomic, meso, or continuum scale.
    (see lammps.org)

    The presented abstract run-only class checks the work of LAMMPS.
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

    num_tasks_per_node = required

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
