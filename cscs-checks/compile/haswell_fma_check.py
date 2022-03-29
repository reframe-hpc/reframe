# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HaswellFmaCheck(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.descr = 'check for avx2 instructions'
        self.valid_systems = ['dom:login', 'daint:login']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi',
                                    'PrgEnv-nvidia']
        self.modules = ['craype-haswell']

        self.sourcesdir = 'src/haswell_fma'
        self.build_system = 'Make'
        self.build_system.cflags = ['-O3', '-S']
        self.build_system.cxxflags = ['-O3', '-S']
        self.build_system.fflags = ['-O3', '-S']
        self.sanity_patterns = sn.all([
            sn.assert_found(r'vfmadd', 'vectorize_fma_c.s'),
            sn.assert_found(r'vfmadd', 'vectorize_fma_cplusplus.s'),
            sn.assert_found(r'vfmadd', 'vectorize_fma_ftn.s'),
            sn.assert_not_found('warning|WARNING', self.stderr)
        ])

        self.maintainers = ['AJ', 'CB']
        self.tags = {'production', 'craype'}

    @run_before('compile')
    def setflags(self):
        if self.current_environ.name == 'PrgEnv-cray':
            self.build_system.cflags = ['-Ofast', '-S']
            self.build_system.cxxflags = ['-Ofast', '-S']
