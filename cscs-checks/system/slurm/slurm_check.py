# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn

# TODO: sinfo, do we want to check if the normal, long, debug, etc... partitions are present?
# TODO: scontrol, do we want to scontrol something specific?
@rfm.parameterized_test(['squeue'],
                        ['sacct'],
                        ['sinfo'],
                        ['scontrol'])
class SlurmCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'File system slurm test base'
        # TODO: test from cn as well
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
            'daint:login': {
                'real_time': (0.1, None, 0.1, 's')
            },
            'dom:login': {
                'real_time': (0.1, None, 0.1, 's')
            }
        }
        # TODO: system is not always relevant
#        self.reference = {
#            '/project/csstaff/bignamic': {
#                'size': (1000, None, 0.1, 'MB'),
#                'real_time': (5.0, None, 0.1, 's')
#            },
#            '/users/bignamic': {
#                'size': (900, None, 0.1, 'MB'),
#                'real_time': (5.0, None, 0.1, 's')
#            },
#            '/scratch/snx3*/bignamic': {
#                'size': (900, None, 0.1, 'MB'),
#                'real_time': (5.0, None, 0.1, 's')
#            }
#        }
        self.executable = 'time ' + variant
        if variant == 'sacct':
            self.executable_opts = ['-a']
        elif variant == 'scontrol':
            self.executable_opts = ['show partitions']
        
        self.postrun_cmds = ['echo $?']
        self.tags = {'ops', 'diagnostic'}
        self.maintainers = ['CB']
