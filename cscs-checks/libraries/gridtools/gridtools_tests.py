# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['vertical_advection_dycore_naive'],
                        ['vertical_advection_dycore_mc'],
                        ['simple_hori_diff_naive'], ['simple_hori_diff_mc'],
                        ['vertical_advection_dycore_cuda'],
                        ['simple_hori_diff_cuda'])
class GridToolsCheck(rfm.RegressionTest):
    def __init__(self, variant):
        # Check if this is a device check
        self.descr = 'GridTools test base'
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['CMake', 'Boost']
        is_cuda_test = 'cuda' in variant
        if is_cuda_test:
            self.modules.append('craype-accel-nvidia60')

        self.sourcesdir = 'https://github.com/GridTools/gridtools.git'
        self.build_system = 'CMake'
        self.build_system.config_opts = [
            '-DBoost_NO_BOOST_CMAKE="true"',
            '-DCMAKE_BUILD_TYPE:STRING=Release',
            '-DBUILD_SHARED_LIBS:BOOL=ON',
            '-DGT_GCL_ONLY:BOOL=OFF',
            '-DCMAKE_CXX_COMPILER=CC',
            '-DGT_USE_MPI:BOOL=OFF',
            '-DGT_SINGLE_PRECISION:BOOL=OFF',
            '-DGT_ENABLE_PERFORMANCE_METERS:BOOL=ON',
            '-DGT_TESTS_ICOSAHEDRAL_GRID:BOOL=OFF',
            '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DBOOST_ROOT=$BOOST_ROOT',
            '-DGT_ENABLE_PYUTILS=OFF',
            '-DGT_TESTS_REQUIRE_FORTRAN_COMPILER=ON',
            '-DGT_TESTS_REQUIRE_C_COMPILER=ON',
            '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']

        if is_cuda_test:
            self.build_system.config_opts += [
                '-DGT_ENABLE_BACKEND_X86:BOOL=OFF',
                '-DGT_ENABLE_BACKEND_NAIVE:BOOL=OFF',
                '-DGT_ENABLE_BACKEND_MC=OFF',
                '-DGT_ENABLE_BACKEND_CUDA:BOOL=ON',
                '-DCUDA_ARCH:STRING=sm_60',
                '-DCMAKE_CUDA_HOST_COMPILER:STRING=CC'
            ]
        else:
            self.build_system.config_opts += [
                '-DGT_ENABLE_BACKEND_X86:BOOL=ON',
                '-DGT_ENABLE_BACKEND_NAIVE:BOOL=ON',
                '-DGT_ENABLE_BACKEND_MC=ON',
                '-DGT_ENABLE_BACKEND_CUDA:BOOL=OFF'
            ]

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        if is_cuda_test:
            self.num_gpus_per_node = 1
            self.num_tasks = 1
        else:
            self.valid_systems += ['daint:mc', 'dom:mc']
            self.num_gpus_per_node = 0
            self.num_tasks = 1

        self.sanity_patterns = sn.assert_found(r'PASSED', self.stdout)
        self.perf_patterns = {
            'wall_time': sn.extractsingle(r'(?P<timer>\w+) ms total',
                                          self.stdout, 'timer', int)
        }
        self.build_system.max_concurrency = 2

        self.variant_data = {
            'vertical_advection_dycore_naive': {
                'executable_opts': ['150', '150', '150'],
                'reference': {
                    'daint:mc': {
                        'wall_time': (3400, None, 0.1, 'ms')
                    },
                    'daint:gpu': {
                        'wall_time': (3800, None, 0.1, 'ms')
                    },
                    'dom:mc': {
                        'wall_time': (3400, None, 0.1, 'ms')
                    },
                    'dom:gpu': {
                        'wall_time': (3800, None, 0.1, 'ms')
                    },
                    '*': {
                        'wall_time': (0, None, None, 'ms')
                    }
                }
            },
            'vertical_advection_dycore_mc': {
                'executable_opts': ['150', '150', '150'],
                'reference': {
                    'daint:mc': {
                        'wall_time': (3500, None, 0.1, 'ms')
                    },
                    'daint:gpu': {
                        'wall_time': (3700, None, 0.1, 'ms')
                    },
                    'dom:mc': {
                        'wall_time': (3500, None, 0.1, 'ms')
                    },
                    'dom:gpu': {
                        'wall_time': (3700, None, 0.1, 'ms')
                    },
                    '*': {
                        'wall_time': (0, None, None, 'ms')
                    }
                }
            },
            'simple_hori_diff_naive': {
                'executable_opts': ['100', '100', '100'],
                'reference': {
                    'daint:mc': {
                        'wall_time': (3200, None, 0.1, 'ms')
                    },
                    'daint:gpu': {
                        'wall_time': (3700, None, 0.1, 'ms')
                    },
                    'dom:mc': {
                        'wall_time': (3200, None, 0.1, 'ms')
                    },
                    'dom:gpu': {
                        'wall_time': (3700, None, 0.1, 'ms')
                    },
                    '*': {
                        'wall_time': (0, None, None, 'ms')
                    }
                }
            },
            'simple_hori_diff_mc': {
                'executable_opts': ['100', '100', '100'],
                'reference': {
                    'daint:mc': {
                        'wall_time': (3300, None, 0.1, 'ms')
                    },
                    'daint:gpu': {
                        'wall_time': (3700, None, 0.1, 'ms')
                    },
                    'dom:mc': {
                        'wall_time': (3300, None, 0.1, 'ms')
                    },
                    'dom:gpu': {
                        'wall_time': (3700, None, 0.1, 'ms')
                    },
                    '*': {
                        'wall_time': (0, None, None, 'ms')
                    }
                }
            },
            'vertical_advection_dycore_cuda': {
                'executable_opts': ['200', '200', '200'],
                'reference': {
                    'daint:gpu': {
                        'wall_time': (12000, None, 0.1, 'ms')
                    },
                    'dom:gpu': {
                        'wall_time': (12000, None, 0.1, 'ms')
                    },
                    '*': {
                        'wall_time': (0, None, None, 'ms')
                    }
                }
            },
            'simple_hori_diff_cuda': {
                'executable_opts': ['150', '150', '150'],
                'reference': {
                    'daint:gpu': {
                        'wall_time': (19000, None, 0.1, 'ms')
                    },
                    'dom:gpu': {
                        'wall_time': (19000, None, 0.1, 'ms')
                    },
                    '*': {
                        'wall_time': (0, None, None, 'ms')
                    }
                }
            }
        }

        self.build_system.make_opts = [variant]
        self.executable = os.path.join('regression', variant)
        self.executable_opts = self.variant_data[variant]['executable_opts']
        self.reference = self.variant_data[variant]['reference']
        self.tags = {'scs', 'benchmark'}
        self.maintainers = ['CB']
