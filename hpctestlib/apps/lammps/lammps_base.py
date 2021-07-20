# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class LAMMPSBaseCheck(rfm.RunOnlyRegressionTest):
    modules = ['LAMMPS']
    input_file = variable(str)
    num_tasks_per_node = required
    ener_ref = variable(typ.Dict[str, typ.Tuple[float, float]])

    @run_after('setup')
    def set_generic_perf_references(self):
        self.reference.update({'*': {
            self.benchmark: (0, None, None, 'timesteps/s')
        }})

    @run_after('setup')
    def set_perf_patterns(self):
        self.perf_patterns = {
            self.benchmark: sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                             self.stdout, 'perf', float)
        }

    @sanity_function
    def set_sanity_patterns(self):
        energy = sn.extractsingle(
            r'\s+500000(\s+\S+){3}\s+(?P<energy>\S+)\s+\S+\s\n',
            self.stdout, 'energy', float)
        energy_reference = self.ener_ref[self.benchmark][0]
        energy_diff = sn.abs(energy - energy_reference)
        ref_ener_diff = sn.abs(self.ener_ref[self.benchmark][0] *
                               self.ener_ref[self.benchmark][1])
        return sn.all([
            sn.assert_found(r'Total wall time:', self.stdout),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
