# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SlurmCheck(rfm.RunOnlyRegressionTest):
    slurm_command = parameter(['squeue', 'sacct'])

    def __init__(self):
        self.descr = 'Slurm command test'
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['builtin']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'real (?P<real_time>\S+)',
                                          self.stderr, 'real_time', float)
        }
        self.reference = {
            'squeue': {
                'real_time': (0.02, None, 0.1, 's')
            },
            'sacct': {
                'real_time': (0.1, None, 0.1, 's')
            }
        }

        self.executable = 'time -p ' + self.slurm_command

        self.tags = {'ops', 'diagnostic', 'health'}
        self.maintainers = ['CB', 'VH']

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_eq(self.job.exitcode, 0)
