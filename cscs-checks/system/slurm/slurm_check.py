# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


# TODO: sinfo, check if the normal, long, debug, etc... partitions are present?
# TODO: scontrol, do we want to scontrol something specific?
@rfm.simple_test
class SlurmCheck(rfm.RunOnlyRegressionTest):
    slurm_command = parameter(['squeue', 'sacct', 'sinfo', 'scontrol'])

    def __init__(self):
        self.descr = 'Slurm command test'
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['builtin']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_found(r'0', self.stdout)
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'\nreal.+m(?P<real_time>\S+)s',
                                          self.stderr, 'real_time', float)
        }
        self.reference = {
            'squeue': {
                'real_time': (0.02, None, 0.1, 's')
            },
            'sacct': {
                'real_time': (0.1, None, 0.1, 's')
            },
            'sinfo': {
                'real_time': (0.02, None, 0.1, 's')
            },
            'scontrol': {
                'real_time': (0.01, None, 0.1, 's')
            }
        }

        self.executable = 'time ' + self.slurm_command
        if self.slurm_command == 'scontrol':
            self.executable_opts = ['show partitions']

        self.postrun_cmds = ['echo $?']
        self.tags = {'ops', 'diagnostic', 'health'}
        self.maintainers = ['CB']
