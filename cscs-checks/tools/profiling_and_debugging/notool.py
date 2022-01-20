# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class JacobiNoToolHybrid(rfm.RegressionTest):
    lang = parameter(['C++', 'F90'])
    time_limit = '10m'
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'eiger:mc', 'pilatus:mc']
    valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray', 'PrgEnv-gnu',
                           'PrgEnv-intel', 'PrgEnv-pgi', 'PrgEnv-nvidia']
    build_system = 'Make'
    executable = './jacobi'
    num_tasks = 3
    num_tasks_per_node = 3
    num_cpus_per_task = 4
    num_tasks_per_core = 1
    use_multithreading = False
    num_iterations = variable(int, value=100)
    url = variable(str, value='http://github.com/eth-cscs/hpctools')
    maintainers = ['JG', 'MKr']
    tags = {'production'}

    @run_after('init')
    def set_descr_name(self):
        self.descr = f'Jacobi (without tool) {self.lang} check'
        self.name = f'{type(self).__name__}_{self.lang.replace("+", "p")}'

    @run_after('init')
    def remove_buggy_prgenv(self):
        # FIXME: skipping to avoid "Fatal error in PMPI_Init_thread"
        if self.current_system.name in ('eiger', 'pilatus'):
            self.valid_prog_environs.remove('PrgEnv-nvidia')

    @run_before('compile')
    def set_sources_dir(self):
        self.sourcesdir = os.path.join('src', self.lang)

    @run_before('compile')
    def restrict_f90_concurrency(self):
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if self.lang == 'F90':
            self.build_system.max_concurrency = 1

    @run_before('compile')
    def set_env_variables(self):
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'ITERATIONS': str(self.num_iterations),
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PROC_BIND': 'true',
        }

    @run_before('compile')
    def set_flags(self):
        self.prebuild_cmds += ['module list']
        self.prgenv_flags = {
            'PrgEnv-aocc': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-cray': ['-O2', '-g',
                            '-homp' if self.lang == 'F90' else '-fopenmp'],
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp'],
            'PrgEnv-nvidia': ['-O2', '-g', '-mp']
        }
        envname = self.current_environ.name
        # if generic, falls back to -g:
        prgenv_flags = self.prgenv_flags.get(envname, ['-g'])
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags

    @run_before('run')
    def set_prerun_cmds(self):
        if self.current_system.name in {'dom', 'daint', 'eiger', 'pilatus'}:
            # get general info about the environment:
            self.prerun_cmds += ['module list']
            self.prerun_cmds += [
                # cray/aocc compilers version are needed but others won't hurt:
                f'echo CRAY_CC_VERSION=$CRAY_CC_VERSION',
                f'echo CRAY_AOCC_VERSION=$CRAY_AOCC_VERSION',
                f'echo GNU_VERSION=$GNU_VERSION',
                f'echo PGI_VERSION=$PGI_VERSION',
                f'echo INTEL_VERSION=$INTEL_VERSION',
                f'echo INTEL_COMPILER_TYPE=$INTEL_COMPILER_TYPE',
            ]

    @run_before('run')
    def set_postrun_cmds(self):
        readme_str = (
            rf'More debug and performance tools ReFrame checks are '
            rf'available at {self.url}'
        )
        self.postrun_cmds += [f'echo "{readme_str}"']

    @sanity_function
    def assert_success(self):
        envname = self.current_environ.name
        # {{{ extract CCE version to manage compiler versions:
        cce_version = None
        rptf = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        if self.lang == 'C++' and envname == 'PrgEnv-cray':
            cce_version = sn.extractsingle(r'CRAY_CC_VERSION=(\d+)\.\S+', rptf,
                                           1, int)
        # }}}

        # {{{ extract AOCC version to manage compiler versions:
        aocc_version = None
        rptf = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        if self.lang == 'C++' and envname == 'PrgEnv-aocc':
            aocc_version = sn.extractsingle(
                r'CRAY_AOCC_VERSION=(\d+)\.\S+', rptf, 1, int)
        # }}}

        intel_type = sn.extractsingle(r'INTEL_COMPILER_TYPE=(\S*)', rptf, 1)
        # {{{ print(f'intel_type={intel_type}')
        # intel/19.1.1.217        icpc openmp/201611
        # intel/19.1.3.304        icpc openmp/201611
        # intel/2021.2.0          icpc openmp/201611
        # intel-classic/2021.2.0  icpc openmp/201611
        # intel-oneapi/2021.2.0   icpc openmp/201611 = 4.5
        # intel-oneapi/2021.2.0   icpx openmp/201811 = 5.0
        # __INTEL_COMPILER
        # INTEL_VERSION 2021.2.0 INTEL_COMPILER_TYPE ONEAPI      201811
        # INTEL_VERSION 2021.2.0 INTEL_COMPILER_TYPE RECOMMENDED
        # INTEL_VERSION 2021.2.0 INTEL_COMPILER_TYPE CLASSIC     201611
        # }}}
        # OpenMP support varies between compilers:
        #            c++ - f90
        #  aocc - 201511 - 201307
        #   cce - 201511 - 201511
        #   gnu - 201511 - 201511
        # intel - 201811 - 201611
        #   pgi - 201307 - 201307
        #    nv - 201307 - 201307
        openmp_versions = {
            # 'PrgEnv-aocc': {'C++': 201511, 'F90': 201307},
            'PrgEnv-aocc': {
                'C++': 201511 if aocc_version == 2 else 201811,
                'F90': 201307,
            },
            'PrgEnv-cray': {
                'C++': 201511 if cce_version == 10 else 201811,
                'F90': 201511,
            },
            'PrgEnv-gnu': {'C++': 201511, 'F90': 201511},
            'PrgEnv-intel': {
                'C++': 201811 if (intel_type == 'ONEAPI' or
                                  intel_type == 'RECOMMENDED') else 201611,
                'F90': 201611},
            'PrgEnv-pgi': {'C++': 201307, 'F90': 201307},
            'PrgEnv-nvidia': {'C++': 201307, 'F90': 201307}
        }
        found_version = sn.extractsingle(r'OpenMP-\s*(\d+)', self.stdout, 1,
                                         int)
        return sn.all([
            sn.assert_found('SUCCESS', self.stdout),
            sn.assert_eq(found_version, openmp_versions[envname][self.lang])
        ])

    @performance_function('s')
    def elapsed_time(self):
        return sn.extractsingle(
            r'Elapsed Time\s*:\s+(\S+)', self.stdout, 1, float)
