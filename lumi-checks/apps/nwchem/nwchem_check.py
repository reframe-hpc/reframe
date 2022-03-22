# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NwchemCheck(rfm.RunOnlyRegressionTest):
    arch = ['cpu']
    valid_prog_environs = ['builtin', 'cpeGNU']
    modules = ['cray-python','NWChem']
    executable = 'nwchem'
    # Water SCF calculation and geometry optimization in a 6-31g basis
    # https://nwchemgit.github.io/Sample.html
    executable_opts = ['h2o.nw']
    use_multithreading = True
    time_limit = '10m'
    maintainers = ['mszpindler']

    @run_after('init')
    def adapt_description(self):
        self.descr = f'NWChem check ({self.arch})'
        self.tags |= {'maintenance', 'production'}

    @run_after('init')
    def adapt_valid_systems(self):
        self.valid_systems = ['lumi:small']

    @run_after('init')
    def adapt_valid_prog_environs(self):
        if self.current_system.name in ['lumi']:
            self.valid_prog_environs = ['cpeGNU']

    @run_after('init')
    def setup_parallel_run(self):
        self.num_tasks = 64
        self.num_tasks_per_node = 64

    @sanity_function
    def validate_energy(self):
        energy_reference = -75.98399
        return sn.assert_found(
            r'Total SCF energy =([ \t]+(?P<energy_reference>\S+))', self.stdout)

    @performance_function('s')
    def walltime(self):
        return sn.extractsingle(r'wall:\s+(\S+)s\s+.*',self.stdout, 1, float)
