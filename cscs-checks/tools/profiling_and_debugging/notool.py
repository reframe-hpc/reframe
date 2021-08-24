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
    time_limit = '10m'

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
            'PrgEnv-nvidia',
        ]
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
            f'echo INTEL_COMPILER_TYPE=$INTEL_COMPILER_TYPE',
            f'echo CRAY_AOCC_VERSION=$CRAY_AOCC_VERSION',
        ]
        self.reference = {'*': {'elapsed_time': (0, None, None, 's')}}

    @run_before('compile')
    def prgEnv_nvidia_workaround(self):
        # {{{ CRAY_MPICH_VERSION
        # cdt/20.08 cray-mpich/7.7.15 cray, crayclang, gnu, intel, pgi
        # cdt/21.02 cray-mpich/7.7.16 cray, crayclang, gnu, intel, pgi
        # cdt/21.05 cray-mpich/7.7.17 crayclang, gnu, intel, pgi, *nvidia*
        #
        # cpe/21.04 cray-mpich/8.1.4 AOCC, CRAY, CRAYCLANG, GNU, INTEL, NVIDIA
        # cpe/21.05 cray-mpich/8.1.5 AOCC, CRAY, CRAYCLANG, GNU, INTEL, NVIDIA
        # cpe/21.06 cray-mpich/8.1.6 AOCC, CRAY, CRAYCLANG, GNU, INTEL, NVIDIA
        # cpe/21.08 cray-mpich/8.1.8 AOCC, CRAY, CRAYCLANG, GNU, INTEL, NVIDIA
        # }}}
        envname = self.current_environ.name
        sysname = self.current_system.name
        self.cppflags = ''
        if (sysname in ['dom', 'daint'] and envname == 'PrgEnv-nvidia'):
            mpi_version = int(os.getenv('CRAY_MPICH_VERSION').replace('.', ''))
            self.skip_if(
                mpi_version <= 7716,
                (f'PrgEnv-nvidia not supported with cray-mpich<=7.7.16 '
                 f'(CRAY_MPICH_VERSION={mpi_version})'))
            # NOTE: occasionally, the wrapper fails to find the mpich dir
            mpich_pkg_config_path = '$CRAY_MPICH_PREFIX/lib/pkgconfig'
            self.variables = {
                'PKG_CONFIG_PATH': f'$PKG_CONFIG_PATH:{mpich_pkg_config_path}'
            }
            self.cppflags = ('`pkg-config --cflags mpich` '
                             '`pkg-config --libs mpich`')
        elif sysname in ['pilatus', 'eiger']:
            self.skip_if(self.current_environ.name == 'PrgEnv-nvidia', '')

    @run_before('compile')
    def set_flags(self):
        # FIXME: workaround for C4KCUST-308
        self.modules += ['cray-mpich']
        self.prgenv_flags = {
            'PrgEnv-aocc': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-cray': ['-O2', '-g',
                            '-homp' if self.lang == 'F90' else '-fopenmp'],
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp'],
            'PrgEnv-nvidia': ['-O2', '-g', '-mp', self.cppflags]
        }
        envname = self.current_environ.name
        # if generic, falls back to -g:
        prgenv_flags = self.prgenv_flags.get(envname, ['-g'])
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.build_system.ldflags = ['-lm']

    @run_before('compile')
    def alps_fix_aocc(self):
        self.prebuild_cmds += ['module list']
        if self.current_partition.fullname in ['eiger:mc', 'pilatus:mc']:
            self.prebuild_cmds += ['module rm cray-libsci']

    @run_before('sanity')
    def set_sanity(self):
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
        # print(f'intel_type={intel_type}')
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

        # OpenMP support varies between compilers:
        self.openmp_versions = {
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
        self.sanity_patterns = sn.all(
            [
                sn.assert_found('SUCCESS', self.stdout),
                sn.assert_eq(found_version,
                             self.openmp_versions[envname][self.lang]),
            ]
        )
