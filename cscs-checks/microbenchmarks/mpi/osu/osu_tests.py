# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import reframe as rfm

from hpctestlib.microbenchmarks.mpi.osu import (build_osu_benchmarks,
                                                osu_build_run)


class cscs_build_osu_benchmarks(build_osu_benchmarks):
    build_type = parameter(['cpu', 'cuda'])

    @run_after('init')
    def setup_modules(self):
        if self.build_type != 'cuda':
            return

        if self.current_system.name in ('daint', 'dom'):
            self.modules = ['cudatoolkit/21.3_11.2']
        elif self.current_system.name in ('arolla', 'tsa'):
            self.modules = ['cuda/10.1.243']
            self.build_system.ldflags = ['-L$EBROOTCUDA/lib64',
                                         '-lcudart', '-lcuda']


cscs_cpu_variant = cscs_build_osu_benchmarks.get_variant_nums(build_type='cpu')
cscs_cuda_variant = cscs_build_osu_benchmarks.get_variant_nums(
    build_type='cuda'
)


class cscs_osu_benchmarks(osu_build_run):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'eiger:mc', 'pilatus:mc', 'arolla:cn', 'tsa:cn']
    valid_prog_environs = ['PrgEnv-gnu']
    exclusive_access = True
    tags = {'production', 'benchmark', 'craype'}
    maintainers = ['@rsarm', '@vkarak']


class cscs_osu_pt2pt_benchmarks(cscs_osu_benchmarks):
    benchmark_info = parameter([
        ('mpi.pt2pt.osu_bw', 'bandwidth'),
        ('mpi.pt2pt.osu_latency', 'latency')
    ], fmt=lambda x: x[0], loggable=True)


# CSCS test specilizations
#
# NOTE: Normally, we would only need two specializations: one for the P2P
# tests and one for the collectives, the reason being simply that we want to
# further parametrize the collective benchmarks. However, we need to
# specialize also depending on the `build_type`. The reason for this is that
# we can't access the `build_type` parameter of the `osu_binaries` fixture,
# until after the setup stage, when the `osu_binaries` fixture is resolved.
# This is a limitation of the framework, which although parametersises a test
# based on a parameterized fixture, there is no straightforward way to
# retrieve the parameter of a fixture before the setup stage.


@rfm.simple_test
class cscs_osu_pt2pt_cpu_check(cscs_osu_pt2pt_benchmarks):
    '''Non CUDA-aware benchmark versions'''

    osu_binaries = fixture(cscs_build_osu_benchmarks, scope='environment',
                           variants=cscs_cpu_variant)
    allref = {
        'mpi.pt2pt.osu_bw': {
            'daint:gpu': {
                'bandwidth': (9481.0, -0.10, None, 'MB/s')
            },
            'daint:mc': {
                'bandwidth': (8507, -0.15, None, 'MB/s')
            },
            'dom:gpu': {
                'bandwidth': (9476.3, -0.05, None, 'MB/s')
            },
            'dom:mc': {
                'bandwidth': (9528.0, -0.20, None, 'MB/s')
            },
            'eiger:mc': {
                'bandwidth': (12240.0, -0.10, None, 'MB/s')
            },
            'pilatus:mc': {
                'bandwidth': (12240.0, -0.10, None, 'MB/s')
            }
        },
        'mpi.pt2pt.osu_latency': {
            'daint:gpu': {
                'latency': (1.40, None, 0.80, 'us')
            },
            'daint:mc': {
                'latency': (1.61, None, 0.70, 'us')
            },
            'dom:gpu': {
                'latency': (1.138, None, 0.10, 'us')
            },
            'dom:mc': {
                'latency': (1.47, None, 0.10, 'us')
            },
            'eiger:mc': {
                'latency': (2.33, None, 0.15, 'us')
            },
            'pilatus:mc': {
                'latency': (2.33, None, 0.15, 'us')
            }
        }
    }

    @run_after('init')
    def setup_reference(self):
        self.reference = self.allref.get(self.benchmark_info[0], {})


@rfm.simple_test
class cscs_osu_pt2pt_cuda_check(cscs_osu_pt2pt_benchmarks):
    '''CUDA-aware benchmark versions'''

    valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
    osu_binaries = fixture(cscs_build_osu_benchmarks, scope='environment',
                           variants=cscs_cuda_variant)
    device_buffers = 'cuda'
    num_gpus_per_node  = 1
    allref = {
        'mpi.pt2pt.osu_bw': {
            'daint:gpu': {
                'bandwidth': (8560, -0.10, None, 'MB/s')
            },
            'dom:gpu': {
                'bandwidth': (8813.09, -0.05, None, 'MB/s')
            },
        },
        'mpi.pt2pt.osu_latency': {
            'daint:gpu': {
                'latency': (6.82, None, 0.50, 'us')
            },
            'dom:gpu': {
                'latency': (5.56, None, 0.1, 'us')
            },
        }
    }

    @run_after('init')
    def setup_daint(self):
        if self.current_system.name in ('daint', 'dom'):
            self.valid_prog_environs = ['PrgEnv-nvidia']
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}


@rfm.simple_test
class cscs_osu_collective_check(cscs_osu_benchmarks):
    benchmark_info = parameter([
        ('mpi.collective.osu_alltoall', 'latency'),
        ('mpi.collective.osu_allreduce', 'latency'),
    ], fmt=lambda x: x[0], loggable=True)
    num_nodes = parameter([6, 16])
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    valid_systems = ['daint:gpu', 'daint:mc']
    allref = {
        'mpi.collective.osu_allreduce': {
            6: {
                'dom:gpu': {
                    'latency': (5.67, None, 0.05, 'us')
                },
                'daint:gpu': {
                    'latency': (8.66, None, 0.85, 'us')
                },
                'daint:mc': {
                    'latency': (10.90, None, 1.90, 'us')
                }
            },
            16: {
                'daint:gpu': {
                    'latency': (13.62, None, 1.16, 'us')
                },
                'daint:mc': {
                    'latency': (19.07, None, 1.64, 'us')
                }
            }
        },
        'mpi.collective.osu_alltoall': {
            6: {
                'dom:gpu': {
                    'latency': (8.23, None, 0.1, 'us')
                },
                'daint:gpu': {
                    'latency': (20.73, None, 2.0, 'us')
                },
                'dom:mc': {
                    'latency': (0, None, None, 'us')
                },
                'daint:mc': {
                    'latency': (0, None, None, 'us')
                }
            },
            16: {
                'daint:gpu': {
                    'latency': (0, None, None, 'us')
                },
                'daint:mc': {
                    'latency': (0, None, None, 'us')
                }
            }
        }
    }
    osu_binaries = fixture(cscs_build_osu_benchmarks, scope='environment',
                           variants=cscs_cpu_variant)

    @run_after('init')
    def setup_test(self):
        self.num_tasks = self.num_nodes
        if self.num_nodes == 6:
            self.valid_systems += ['dom:gpu', 'dom:mc']

        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.num_nodes]
