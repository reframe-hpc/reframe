# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class AmberBaseCheck(rfm.RunOnlyRegressionTest):
    modules = ['Amber']
    input_file = variable(str)
    ouput_file = variable(str)
    num_tasks_per_node = required
    ener_ref = variable(typ.Dict[str, typ.Tuple[int, float]])
    benchmark = parameter([
        # NVE simulations
        'Cellulose_production_NVE',
        'FactorIX_production_NVE',
        'JAC_production_NVE_4fs',
        'JAC_production_NVE',
    ])

    @run_after('init')
    def download_files(self):
        self.prerun_cmds = [
            # cannot use wget because it is not installed on eiger
            f'curl -LJO https://github.com/victorusu/amber_benchmark_suite'
            f'/raw/main/amber_16_benchmark_suite/PME/{self.benchmark}.tar.bz2',
            f'tar xf {self.benchmark}.tar.bz2'
        ]

    @run_after('setup')
    def set_executable_opts(self):
        self.executable_opts = ['-O',
                                '-i', self.input_file,
                                '-o', self.output_file]
        self.keep_files = [self.output_file]

    @run_after('setup')
    def set_sanity_patterns(self):
        energy = sn.extractsingle(r' Etot\s+=\s+(?P<energy>\S+)',
                                  self.output_file, 'energy', float, item=-2)
        energy_reference = self.ener_ref[self.benchmark][0]
        energy_diff = sn.abs(energy - energy_reference)
        ref_ener_diff = sn.abs(self.ener_ref[self.benchmark][0] *
                               self.ener_ref[self.benchmark][1])
        self.sanity_patterns = sn.all([
            sn.assert_found(r'Final Performance Info:', self.output_file),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])

    @run_after('setup')
    def set_generic_perf_references(self):
        self.reference.update({'*': {
            self.benchmark: (0, None, None, 'ns/day')
        }})

    @run_after('setup')
    def set_perf_patterns(self):
        self.perf_patterns = {
            self.benchmark: sn.extractsingle(r'ns/day =\s+(?P<perf>\S+)',
                                             self.output_file, 'perf',
                                             float, item=1)
        }
