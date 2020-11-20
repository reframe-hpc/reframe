# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class FileSystemCommandCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'File system sanity test base'
        self.valid_systems = ['daint:login', 'dom:login'] # TODO: test from cn as well
        self.valid_prog_environs = ['builtin']
        self.num_tasks = 1
        self.num_tasks_per_node = 1

        self.sanity_patterns = sn.assert_found(r'real \d*\.?\d*\n0', self.stdout)
#        self.sanity_patterns = sn.assert_found(r'real.+\nuser.+\nsys.+\n0', self.stdout)
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'real (?P<time>\d*\.?\d*)\n0',
                                          self.stdout, 'timer', float)
        }
        self.postrun_cmds = ['echo $?']
        self.tags = {'ops', 'diagnostic'}
        self.maintainers = ['CB']


@rfm.parameterized_test(['scratch_snx1600'],
                        ['scratch_snx3000'])
class FileSystemChangeDirCheck(FileSystemCommandCheck):  
    def __init__(self, variant):
        super().__init__()
        self.descr = 'Change directory to scratch test' 
        self.variant_data = {
            'scratch_snx1600': {
                'executable_opts': ['/scratch/snx1600'],
                'reference': {
                    'daint:login': {
                        'wall_time': (0.1, None, 0.1, 's')
                    },
                    'daint:login': {
                        'wall_time': (0.1, None, 0.1, 's')
                    }
                }
            },
            'scratch_snx3000': {
                'executable_opts': ['/scratch/snx3000'],
                'reference': {
                    'daint:login': {
                        'wall_time': (0.1, None, 0.1, 's')
                    },
                    'daint:login': {
                        'wall_time': (0.1, None, 0.1, 's')
                    }
                }
            }
        }

        self.executable = '/usr/bin/time -f "real %e" timeout -k 1 5 cd'
        self.executable_opts = self.variant_data[variant]['executable_opts']
        self.reference = self.variant_data[variant]['reference']
