# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import ast
import os
import reframe as rfm
import reframe.utility.sanity as sn


class GridToolsBuildCheck(rfm.CompileOnlyRegressionTest):
    valid_prog_environs = ['builtin']
    modules = ['CMake', 'Boost']
    valid_systems = ['daint:gpu', 'dom:gpu']

    sourcesdir = 'https://github.com/GridTools/gridtools.git'
    build_system = 'CMake'
    postbuild_cmds = ['ls tests/regression/']
    tags = {'scs', 'benchmark'}
    maintainers = ['CB']

    @rfm.run_before('compile')
    def set_build_options(self):
        self.build_system.config_opts = [
            '-DCMAKE_BUILD_TYPE=Debug',
            '-DCMAKE_CXX_FLAGS=-std=c++14',
            '-DCMAKE_CXX_COMPILER=CC',
            '-DCMAKE_C_COMPILER=cc',
            '-DCMAKE_Fortran_COMPILER=ftn',
            '-DGT_TESTS_REQUIRE_FORTRAN_COMPILER=ON',
            '-DGT_TESTS_REQUIRE_C_COMPILER=ON',
            '-DGT_TESTS_REQUIRE_OpenMP="ON"'
        ]
        self.build_system.flags_from_environ = False
        self.build_system.make_opts = ['perftests']
        self.build_system.max_concurrency = 8

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'perftest',
                                               self.stdout)


@rfm.simple_test
class GridToolsCPUBuildCheck(GridToolsBuildCheck):
    descr = 'GridTools CPU build test'

    @rfm.run_after('init')
    def add_valid_systems(self):
        self.valid_systems += ['daint:mc', 'dom:mc']

    @rfm.run_before('compile')
    def set_cpu_build_option(self):
        self.build_system.config_opts += [
            '-DGT_TESTS_REQUIRE_GPU="OFF"'
        ]


@rfm.simple_test
class GridToolsGPUBuildCheck(GridToolsBuildCheck):
    descr = 'GridTools GPU build test'

    @rfm.run_before('compile')
    def set_env(self):
        if self.current_system.name == 'dom':
            self.modules += [
                'cudatoolkit'
                'cdt-cuda'
            ]
        else:
            self.modules.append('cudatoolkit')

    @rfm.run_before('compile')
    def set_gpu_build_options(self):
        self.build_system.config_opts += [
            '-DGT_CUDA_ARCH=sm_60',
            '-DGT_TESTS_REQUIRE_GPU="ON"'
        ]


class GridToolsRunCheck(rfm.RunOnlyRegressionTest):
    valid_prog_environs = ['builtin']
    modules = ['CMake', 'Boost']
    valid_systems = ['daint:gpu', 'dom:gpu']
    num_tasks = 1

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'PASSED', self.stdout)
        literal_eval = sn.sanity_function(ast.literal_eval)
        self.perf_patterns = {
            'wall_time': sn.avg(literal_eval(
                sn.extractsingle(r'"series" : \[(?P<wall_times>.+)\]',
                                 self.stdout, 'wall_times')))
        }


@rfm.simple_test
class GridToolsCPURunCheck(GridToolsRunCheck):
    variant = parameter(['horizontal_diffusion/cpu_kfirst_double',
                        'horizontal_diffusion/cpu_ifirst_double'])
    descr = 'GridTools CPU run test'
    num_gpus_per_node = 0
    variant_data = {
        'horizontal_diffusion/cpu_kfirst_double': {
            'reference': {
                'daint:mc': {
                    'wall_time': (11.0, None, 0.05, 's')
                },
                'daint:gpu': {
                    'wall_time': (1.09, None, 0.05, 's')
                },
                'dom:mc': {
                    'wall_time': (11.0, None, 0.05, 's')
                },
                'dom:gpu': {
                    'wall_time': (1.09, None, 0.05, 's')
                }
            }
        },
        'horizontal_diffusion/cpu_ifirst_double': {
            'reference': {
                'daint:mc': {
                    'wall_time': (9.0, None, 0.05, 's')
                },
                'daint:gpu': {
                    'wall_time': (1.0, None, 0.05, 's')
                },
                'dom:mc': {
                    'wall_time': (9.0, None, 0.05, 's')
                },
                'dom:gpu': {
                    'wall_time': (1.0, None, 0.05, 's')
                }
            }
        }
    }
    tags = {'scs', 'benchmark'}
    maintainers = ['CB']

    @rfm.require_deps
    def set_executable(self, GridToolsCPUBuildCheck):
        self.executable = os.path.join(
            GridToolsCPUBuildCheck().stagedir,
            'tests', 'regression', 'perftests'
        )

    @rfm.run_after('init')
    def add_valid_systems(self):
        self.valid_systems += ['daint:mc', 'dom:mc']

    @rfm.run_after('init')
    def set_dependencies(self):
        self.depends_on('GridToolsCPUBuildCheck')

    @rfm.run_before('run')
    def set_executable_opts(self):
        self.executable_opts = ['256', '256', '80', '3',
                                f'--gtest_filter={self.variant}*']

    @rfm.run_before('sanity')
    def set_performance_reference(self):
        self.reference = self.variant_data[self.variant]['reference']


@rfm.simple_test
class GridToolsGPURunCheck(GridToolsRunCheck):
    variant = parameter(['horizontal_diffusion/gpu_double',
                         'horizontal_diffusion/gpu_horizontal_double'])
    descr = 'GridTools GPU run test'
    num_gpus_per_node = 1
    variant_data = {
        'horizontal_diffusion/gpu_double': {
            'reference': {
                'daint:gpu': {
                    'wall_time': (0.004, None, 0.05, 's')
                },
                'dom:gpu': {
                    'wall_time': (0.004, None, 0.05, 's')
                }
            }
        },
        'horizontal_diffusion/gpu_horizontal_double': {
            'reference': {
                'daint:gpu': {
                    'wall_time': (0.003, None, 0.05, 's')
                },
                'dom:gpu': {
                    'wall_time': (0.003, None, 0.05, 's')
                }
            }
        }
    }
    tags = {'scs', 'benchmark'}
    maintainers = ['CB']

    @rfm.require_deps
    def set_executable(self, GridToolsGPUBuildCheck):
        self.executable = os.path.join(
            GridToolsGPUBuildCheck().stagedir,
            'tests', 'regression', 'perftests')

    @rfm.run_after('init')
    def set_dependencies(self):
        self.depends_on('GridToolsGPUBuildCheck')

    @rfm.run_before('compile')
    def modules_update(self):
        self.modules.append('cudatoolkit')

    @rfm.run_before('run')
    def set_executable_opts(self):
        self.executable_opts = ['512', '512', '160', '3',
                                f'--gtest_filter={self.variant}*']

    @rfm.run_before('sanity')
    def set_performance_reference(self):
        self.reference = self.variant_data[self.variant]['reference']
