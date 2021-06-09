# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class JacobiNoToolHybrid(rfm.RegressionTest):
    lang = parameter(['C++', 'F90'])

    def __init__(self):
        super().__init__()
        self.descr = f'Jacobi (without tool) {self.lang} check'
        self.name = f'{type(self).__name__}_{self.lang.replace("+", "p")}'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'eiger:mc', 'pilatus:mc']
        self.valid_prog_environs = [
            'PrgEnv-aocc',
            'PrgEnv-cray',
            'PrgEnv-gnu',
            'PrgEnv-intel',
            'PrgEnv-pgi',
        ]
        self.prgenv_flags = {
            'PrgEnv-aocc': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-cray': ['-O2', '-g',
                            '-homp' if self.lang == 'F90' else '-fopenmp'],
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp']
        }
        self.sourcesdir = os.path.join('src', self.lang)
        self.build_system = 'Make'
        self.executable = './jacobi'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if self.lang == 'F90':
            self.build_system.max_concurrency = 1

        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.num_iterations = 100
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.maintainers = ['JG', 'MKr']
        self.tags = {'production'}
        url = 'http://github.com/eth-cscs/hpctools'
        readme_str = (
            rf'More debug and performance tools ReFrame checks are'
            rf' available at {url}'
        )
        self.postrun_cmds += [f'echo "{readme_str}"']
        if self.current_system.name in {'dom', 'daint', 'eiger', 'pilatus'}:
            # get general info about the environment:
            self.prerun_cmds += ['module list']
        self.perf_patterns = {
            'elapsed_time': sn.extractsingle(
                r'Elapsed Time\s*:\s+(\S+)', self.stdout, 1, float
            )
        }
        self.prerun_cmds += [
            # only cray compiler version is really needed but this won't hurt:
            f'echo CRAY_CC_VERSION=$CRAY_CC_VERSION',
            f'echo GNU_VERSION=$GNU_VERSION',
            f'echo PGI_VERSION=$PGI_VERSION',
            f'echo INTEL_VERSION=$INTEL_VERSION',
            f'echo CRAY_AOCC_VERSION=$CRAY_AOCC_VERSION',
        ]
        self.reference = {'*': {'elapsed_time': (0, None, None, 's')}}
        if self.lang == 'C++':
            self.reference_lang = (0.38, -0.6, None, 's')
        elif self.lang == 'F90':
            self.reference_lang = (0.17, -0.6, None, 's')

    @run_before('compile')
    def set_flags(self):
        envname = self.current_environ.name
        # if generic, falls back to -g:
        prgenv_flags = self.prgenv_flags.get(envname, ['-g'])
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.build_system.ldflags = ['-lm']

    @run_before('compile')
    def set_cuda_cdt(self):
        if self.current_partition.name == 'gpu':
            self.modules += ['cdt-cuda']

    @run_before('compile')
    def alps_fix_aocc(self):
        if self.current_partition.fullname in ['eiger:mc', 'pilatus:mc']:
            self.prebuild_cmds += ['module rm cray-libsci']

    @run_before('sanity')
    def set_sanity(self):
        envname = self.current_environ.name
        # CCE specific:
        cce_version = None
        if self.lang == 'C++' and envname == 'PrgEnv-cray':
            rptf = os.path.join(self.stagedir, sn.evaluate(self.stdout))
            cce_version = sn.extractsingle(r'CRAY_CC_VERSION=(\d+)\.\S+', rptf,
                                           1, int)

        # OpenMP support varies between compilers:
        self.openmp_versions = {
            'PrgEnv-aocc': {'C++': 201511, 'F90': 201307},
            'PrgEnv-cray': {
                'C++': 201511 if cce_version == 10 else 201811,
                'F90': 201511,
            },
            'PrgEnv-gnu': {'C++': 201511, 'F90': 201511},
            'PrgEnv-intel': {'C++': 201611, 'F90': 201611},
            'PrgEnv-pgi': {'C++': 201307, 'F90': 201307}
        }
        found_version = sn.extractsingle(r'OpenMP-\s*(\d+)', self.stdout, 1,
                                         int)
        self.sanity_patterns = sn.all(
            [
                sn.assert_found('SUCCESS', self.stdout),
                sn.assert_eq(found_version,
                             self.openmp_versions[envname][self.lang]),
            ]
        )

    @run_before('sanity')
    def set_reference(self):
        if self.current_system.name in {'dom', 'daint'}:
            self.reference['*:elapsed_time'] = self.reference_lang
