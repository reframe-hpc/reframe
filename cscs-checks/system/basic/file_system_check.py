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
        self.postrun_cmds = ['echo $?']
        self.tags = {'ops', 'diagnostic'}
        self.maintainers = ['CB']


@rfm.parameterized_test(['/scratch/snx1*'],
                        ['/scratch/snx3*'])
class FileSystemChangeDirCheck(FileSystemCommandCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = 'Change directory to scratch test'
        self.reference = {
            'daint:login': {
                'real_time': (0.1, None, 0.1, 's')
            },
            'dom:login': {
                'real_time': (0.1, None, 0.1, 's')
            }
        }
        self.executable = 'time cd'
        self.executable_opts = [variant]


@rfm.parameterized_test(['/scratch/snx1*'],
                        ['/scratch/snx3*'])
class FileSystemLsDirCheck(FileSystemCommandCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = 'ls of directory in scratch test'
        self.reference = {
            'daint:login': {
                'real_time': (0.1, None, 0.1, 's')
            },
            'dom:login': {
                'real_time': (0.1, None, 0.1, 's')
            }
        }
        self.executable = 'time ls'
        self.executable_opts = [variant]

                       
@rfm.parameterized_test(['/project/csstaff/bignamic'],#TODO: can we parametrize the user?
                        ['/users/bignamic'],
                        ['/scratch/snx1*/bignamic'],
                        ['/scratch/snx3*/bignamic'])#TODO: what is the correct way to do this?
class FileSystemDuDirCheck(FileSystemCommandCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = 'du of directory'

        # TODO: is it possible to append a pattern?
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'\nreal.+m(?P<real_time>\S+)s',
                                          self.stderr, 'real_time', float),
#            'size' : sn.extractsingle(r'(?P<size>\S+)\s+%s'%variant,
#                                      self.stdout, 'size', float)
            'size' : sn.extractsingle(r'(?P<size>\S+).+/bignamic',# TODO: this should be solved with parametrized user 
                                      self.stdout, 'size', float)
        }
        
        # TODO: system is not always relevant 
        self.reference = {
            '/project/csstaff/bignamic': {
                'size': (1000, None, 0.1, 'MB'),
                'real_time': (5.0, None, 0.1, 's')                
            },
            '/users/bignamic': {
                'size': (900, None, 0.1, 'MB'),
                'real_time': (5.0, None, 0.1, 's')
            },
            '/scratch/snx3*/bignamic': {
                'size': (900, None, 0.1, 'MB'),
                'real_time': (5.0, None, 0.1, 's')
            }
        }        
        self.executable = 'time du -mhs --block-size=1M'
        self.executable_opts = [variant]


@rfm.parameterized_test(['/scratch/snx1*/bignamic'],#TODO: can we parametrize the user?
                        ['/scratch/snx3000/bignamic'])#TODO: what is the correct way to do this?
class FileSystemTouchFileCheck(FileSystemCommandCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = 'touch of file'
        self.executable = 'time touch'
        self.test_file = variant + '/reframe_touch_test_file'
        self.executable_opts = [self.test_file]

    @rfm.run_after('run')
    def delete_test_file(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)    
