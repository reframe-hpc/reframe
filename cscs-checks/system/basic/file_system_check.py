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

        self.sanity_patterns = sn.assert_found(r'0', self.stdout)
        self.perf_patterns = {
            'real': sn.extractsingle(r'\nreal.+m(?P<real>\S+)s',
                                     self.stderr, 'real', float)
        }
        self.postrun_cmds = ['echo $?']
        self.tags = {'ops', 'diagnostic'}
        self.maintainers = ['CB']


@rfm.parameterized_test(['scratch/snx1600'],
                        ['scratch/snx3000'])
class FileSystemChangeDirCheck(FileSystemCommandCheck):  
    def __init__(self, variant):
        super().__init__()
        self.descr = 'Change directory to scratch test' 
        self.variant_data = {
            'scratch/snx1600': {
                'executable_opts': ['/scratch/snx1600'],
                'reference': {
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    },
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    }
                }
            },
            'scratch/snx3000': {
                'executable_opts': ['/scratch/snx3000'],
                'reference': {
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    },
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    }
                }
            }
        }

        self.executable = 'time cd'
        self.executable_opts = self.variant_data[variant]['executable_opts']
        self.reference = self.variant_data[variant]['reference']


@rfm.parameterized_test(['scratch/snx1600'],
                        ['scratch/snx3000'])
class FileSystemLsDirCheck(FileSystemCommandCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = 'ls of directory in scratch test'
        self.variant_data = {
            'scratch/snx1600': {
                'executable_opts': ['/scratch/snx1600'],
                'reference': {
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    },
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    }
                }
            },
            'scratch/snx3000': {
                'executable_opts': ['/scratch/snx3000'],
                'reference': {
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    },
                    'daint:login': {
                        'real': (0.1, None, 0.1, 's')
                    }
                }
            }
        }

        self.executable = 'time ls'
        self.executable_opts = self.variant_data[variant]['executable_opts']
        self.reference = self.variant_data[variant]['reference']
