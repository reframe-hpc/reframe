# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ReFrame CSCS settings
#

import reframe.utility.osext as osext


site_configuration = {
    'systems': [
        {
            'name': 'ault',
            'descr': 'Ault TDS',
            'hostnames': ['ault'],
            'modules_system': 'lmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': ['builtin', 'PrgEnv-gnu'],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'a64fx',
                    'scheduler': 'slurm',
                    'access': ['-pa64fx'],
                    'environs': ['builtin', 'PrgEnv-gnu'],
                    'descr': 'Fujitsu A64FX CPUs',
                    'max_jobs': 100,
                    'launcher': 'srun'
                },
                {
                    'name': 'amda100',
                    'scheduler': 'slurm',
                    'access': ['-pamda100'],
                    'environs': ['builtin', 'PrgEnv-gnu'],
                    'descr': 'AMD Naples 32c + 4x NVIDIA A100',
                    'max_jobs': 100,
                    'launcher': 'srun'
                },
                {
                    'name': 'amdv100',
                    'scheduler': 'slurm',
                    'access': ['-pamdv100'],
                    'environs': ['builtin', 'PrgEnv-gnu'],
                    'descr': 'AMD Naples 32c + 2x NVIDIA V100',
                    'max_jobs': 100,
                    'launcher': 'srun'
                },
                {
                    'name': 'amdvega',
                    'scheduler': 'slurm',
                    'access': ['-pamdvega'],
                    'environs': ['builtin', 'PrgEnv-gnu'],
                    'descr': 'AMD Naples 32c + 3x AMD GFX900',
                    'max_jobs': 100,
                    'launcher': 'srun'
                },
                {
                    'name': 'intelv100',
                    'scheduler': 'slurm',
                    'access': ['-pintelv100'],
                    'environs': ['builtin', 'PrgEnv-gnu'],
                    'descr': 'Intel Skylake 36c + 4x NVIDIA V100',
                    'max_jobs': 100,
                    'launcher': 'srun'
                },
                {
                    'name': 'intel',
                    'scheduler': 'slurm',
                    'access': ['-pintel'],
                    'environs': ['builtin', 'PrgEnv-gnu'],
                    'descr': 'Intel Skylake 36c',
                    'max_jobs': 100,
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'tave',
            'descr': 'Grand Tave',
            'hostnames': ['tave'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-pgi'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'compute',
                    'scheduler': 'slurm',
                    'environs': [
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-pgi'
                    ],
                    'descr': 'Intel Xeon Phi',
                    'max_jobs': 100,
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'daint',
            'descr': 'Piz Daint',
            'hostnames': ['daint'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'gpu',
                    'time_limit': '10m',
                    'scheduler': 'slurm',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.6.4-daint']
                        }
                    ],
                    'modules': ['daint-gpu'],
                    'access': [
                        f'--constraint=gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'mc',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.6.4-daint']
                        }
                    ],
                    'modules': ['daint-mc'],
                    'access': [
                        f'--constraint=mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Multicore nodes (Broadwell)',
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_gpu',
                    'scheduler': 'slurm',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cgpu',
                        f'--reservation=interact_gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub GPU nodes',
                    'max_jobs': 10,
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_mc',
                    'scheduler': 'slurm',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cmc',
                        f'--reservation=interact_mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub multicore nodes',
                    'max_jobs': 10,
                    'launcher': 'srun'
                },
                {
                    'name': 'xfer',
                    'scheduler': 'slurm',
                    'environs': ['builtin'],
                    'access': [
                        f'--partition=xfer',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'Nordend nodes for internal transfers',
                    'max_jobs': 10,
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'dom',
            'descr': 'Dom TDS',
            'hostnames': ['dom'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'gpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.8.0-daint']
                        }
                    ],
                    'modules': ['daint-gpu'],
                    'access': [
                        f'--constraint=gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'launcher': 'srun',
                    'resources': [
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ]
                },
                {
                    'name': 'mc',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.8.0-daint']
                        }
                    ],
                    'modules': ['daint-mc'],
                    'access': [
                        f'--constraint=mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Multicore nodes (Broadwell)',
                    'max_jobs': 100,
                    'resources': [
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_gpu',
                    'scheduler': 'slurm',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cgpu',
                        f'--reservation=interact_gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub GPU nodes',
                    'max_jobs': 10,
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_mc',
                    'scheduler': 'slurm',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cmc',
                        f'--reservation=interact_mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub multicore nodes',
                    'max_jobs': 10,
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'fulen',
            'descr': 'Fulen',
            'hostnames': [r'fulen-ln\d+'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': ['PrgEnv-gnu'],
                    'descr': 'Login nodes',
                    'max_jobs': 1,
                    'launcher': 'local'
                },
                {
                    'name': 'normal',
                    'scheduler': 'slurm',
                    'environs': ['PrgEnv-gnu'],
                    'descr': 'Compute nodes - default partition',
                    'launcher': 'srun'
                },
                {
                    'name': 'fat',
                    'scheduler': 'slurm',
                    'environs': ['PrgEnv-gnu'],
                    'access': ['--partition fat'],
                    'descr': 'High-memory compute nodes',
                    'launcher': 'srun'
                },
                {
                    'name': 'gpu',
                    'scheduler': 'slurm',
                    'environs': ['PrgEnv-gnu'],
                    'access': ['--partition gpu'],
                    'descr': 'Hybrid compute nodes',
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'arolla',
            'descr': 'Arolla MCH',
            'hostnames': [r'arolla-\w+\d+'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Arolla login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'pn',
                    'scheduler': 'slurm',
                    'access': ['--partition=pn-regression'],
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Arolla post-processing nodes',
                    'max_jobs': 50,
                    'launcher': 'srun'
                },
                {
                    'name': 'cn',
                    'scheduler': 'slurm',
                    'access': ['--partition=cn-regression'],
                    'environs': [
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nompi-nocuda',
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nompi-nocuda'
                    ],
                    'descr': 'Arolla compute nodes',
                    'resources': [
                        {
                            'name': '_rfm_gpu',
                            'options': ['--gres=gpu:{num_gpus_per_node}']
                        }
                    ],
                    'max_jobs': 50,
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'tsa',
            'descr': 'Tsa MCH',
            'hostnames': [r'tsa-\w+\d+'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nocuda',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nocuda',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Tsa login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'pn',
                    'scheduler': 'slurm',
                    'access': ['--partition=pn-regression'],
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nocuda',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nocuda',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Tsa post-processing nodes',
                    'max_jobs': 20,
                    'launcher': 'srun'
                },
                {
                    'name': 'cn',
                    'scheduler': 'slurm',
                    'access': ['--partition=cn-regression'],
                    'environs': [
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nocuda',
                        'PrgEnv-gnu-nompi-nocuda',
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nocuda',
                        'PrgEnv-pgi-nompi-nocuda'
                    ],
                    'descr': 'Tsa compute nodes',
                    'max_jobs': 20,
                    'resources': [
                        {
                            'name': '_rfm_gpu',
                            'options': ['--gres=gpu:{num_gpus_per_node}']
                        }
                    ],
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'eiger',
            'descr': 'Alps Cray EX Supercomputer',
            'hostnames': ['eiger'],
            'modules_system': 'lmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'mc',
                    'descr': 'Multicore nodes (AMD EPYC 7742, 256|512GB/cn)',
                    'scheduler': 'slurm',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.5.3-eiger']
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'max_jobs': 100,
                    'access': ['-Cmc', f'--account={osext.osgroup()}'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_mc',
                    'scheduler': 'slurm',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cmc',
                        f'--reservation=interact',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub GPU nodes',
                    'max_jobs': 10,
                    'launcher': 'srun'
                },
            ]
        },
        {
            'name': 'pilatus',
            'descr': 'Alps Cray EX Supercomputer TDS',
            'hostnames': ['pilatus'],
            'modules_system': 'lmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'mc',
                    'descr': 'Multicore nodes (AMD EPYC 7742, 256|512GB/cn)',
                    'scheduler': 'slurm',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.5.3-eiger']
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'max_jobs': 100,
                    'access': ['-Cmc', f'--account={osext.osgroup()}'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
            ]
        },
        {
            'name': 'generic',
            'descr': 'Generic fallback system',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'environs': ['builtin'],
                    'descr': 'Login nodes',
                    'launcher': 'local'
                }
            ],
            'hostnames': ['.*']
        }
    ],
    'environments': [
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['ault'],
            'modules': ['openmpi'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpif90'
        },
        {
            'name': 'builtin',
            'target_systems': ['ault'],
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
        {
            'name': 'builtin-gcc',
            'target_systems': ['ault'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        },
        {
            'name': 'PrgEnv-pgi-nompi-nocuda',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9-nocuda'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi-nompi-nocuda',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4-nocuda'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi-nompi',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi-nompi',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-pgi',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-pgi-nocuda',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9-nocuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-pgi-nocuda',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4-nocuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-gnu-nocuda',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2-nocuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-gnu-nompi',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        },
        {
            'name': 'PrgEnv-gnu-nompi-nocuda',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2-nocuda'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        },
        {
            'name': 'PrgEnv-aocc',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-aocc']
        },
        {
            'name': 'PrgEnv-cray',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-gnu']
        },
        {
            'name': 'PrgEnv-intel',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-intel']
        },
        {
            'name': 'cpeAMD',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeAMD']
        },
        {
            'name': 'cpeCray',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeCray']
        },
        {
            'name': 'cpeGNU',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeGNU']
        },
        {
            'name': 'cpeIntel',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeIntel']
        },
        {
            'name': 'PrgEnv-cray',
            'modules': ['PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu']
        },
        {
            'name': 'PrgEnv-intel',
            'modules': ['PrgEnv-intel']
        },
        {
            'name': 'PrgEnv-pgi',
            'modules': ['PrgEnv-pgi']
        },
        {
            'name': 'PrgEnv-nvidia',
            'target_systems': ['pilatus'],
            'modules': [
                'PrgEnv-nvidia',
                # FIXME: We should not be forcing a cdt version
                'cpe/21.06'
            ]
        },
        {
            'name': 'PrgEnv-nvidia',
            'target_systems': ['daint', 'dom'],
            'modules': [
                'PrgEnv-nvidia',
            ]
        },
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn'
        },
        {
            'name': 'builtin-gcc',
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        }
    ],
    'logging': [
        {
            'handlers': [
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug2',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                },
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'name': 'reframe.out',
                    'level': 'info',
                    'format': '%(message)s',
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': '%(check_job_completion_time)s|reframe %(version)s|%(check_info)s|jobid=%(check_jobid)s|num_tasks=%(check_num_tasks)s|%(check_perf_var)s=%(check_perf_value)s|ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, u=%(check_perf_upper_thres)s)|%(check_perf_unit)s',   # noqa: E501
                    'datefmt': '%FT%T%:z',
                    'append': True
                },
                {
                    'type': 'httpjson',
                    'url': 'http://httpjson-server:12345/rfm',
                    'level': 'info',
                    'extras': {
                        'facility': 'reframe',
                        'data-version': '1.0',
                    }
                }
            ]
        }
    ],
    'modes': [
        {
            'name': 'maintenance',
            'options': [
                '--unload-module=reframe',
                '--exec-policy=async',
                '--strict',
                '--output=$APPS/UES/$USER/regression/maintenance',
                '--perflogdir=$APPS/UES/$USER/regression/maintenance/logs',
                '--stage=$SCRATCH/regression/maintenance/stage',
                '--report-file=$APPS/UES/$USER/regression/maintenance/reports/maint_report_{sessionid}.json',
                '-Jreservation=maintenance',
                '--save-log-files',
                '--tag=maintenance',
                '--timestamp=%F_%H-%M-%S'
            ]
        },
        {
            'name': 'production',
            'options': [
                '--unload-module=reframe',
                '--exec-policy=async',
                '--strict',
                '--output=$APPS/UES/$USER/regression/production',
                '--perflogdir=$APPS/UES/$USER/regression/production/logs',
                '--stage=$SCRATCH/regression/production/stage',
                '--report-file=$APPS/UES/$USER/regression/production/reports/prod_report_{sessionid}.json',
                '--save-log-files',
                '--tag=production',
                '--timestamp=%F_%H-%M-%S'
            ]
        }
    ],
    'general': [
        {
            'check_search_path': ['checks/'],
            'check_search_recursive': True,
            'remote_detect': True
        }
    ]
}
