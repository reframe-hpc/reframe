# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AllocSpeedTest(rfm.RegressionTest):
    hugepages = parameter(['no', '2M'])
    sourcepath = 'alloc_speed.cpp'
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'eiger:mc', 'pilatus:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    build_system = 'SingleSource'
    maintainers = ['AK', 'VH']
    tags = {'production', 'craype'}

    @run_after('init')
    def set_descr(self):
        self.descr = (f'Time to allocate 4096 MB using {self.hugepages} '
                      f'hugepages')

    @run_after('init')
    def add_valid_systems(self):
        if self.hugepages == 'no':
            self.valid_systems += ['arolla:cn', 'arolla:pn',
                                   'tsa:cn', 'tsa:pn']

    @run_after('init')
    def set_modules(self):
        if self.hugepages != 'no':
            self.modules = [f'craype-hugepages{self.hugepages}']

    @run_before('compile')
    def set_cxxflags(self):
        self.build_system.cxxflags = ['-O3', '-std=c++11']

    @sanity_function
    def assert_4GB(self):
        return sn.assert_found('4096 MB', self.stdout)

    @run_before('performance')
    def set_reference(self):
        sys_reference = {
            'no': {
                'dom:gpu': {
                    'time': (1.22, -0.20, 0.15, 's')
                },
                'dom:mc': {
                    'time': (1.41, -0.20, 0.10, 's')
                },
                'daint:gpu': {
                    'time': (1.22, -0.20, 0.05, 's')
                },
                'daint:mc': {
                    'time': (1.41, -0.20, 0.05, 's')
                },
                'eiger:mc': {
                    'time': (0.12, -0.20, 0.05, 's')
                },
                'pilatus:mc': {
                    'time': (0.12, -0.20, 0.05, 's')
                },
            },
            '2M': {
                'dom:gpu': {
                    'time': (0.11, -0.20, 0.10, 's')
                },
                'dom:mc': {
                    'time': (0.20, -0.20, 0.10, 's')
                },
                'daint:gpu': {
                    'time': (0.11, -0.20, 0.10, 's')
                },
                'daint:mc': {
                    'time': (0.20, -0.20, 0.10, 's')
                },
                'eiger:mc': {
                    'time': (0.06, -0.20, 0.10, 's')
                },
                'pilatus:mc': {
                    'time': (0.06, -0.20, 0.10, 's')
                },
                '*': {
                    'time': (0, None, None, 's')
                }
            },
        }
        self.reference = sys_reference[self.hugepages]

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'4096 MB, allocation time (?P<time>\S+)',
                                self.stdout, 'time', float)
