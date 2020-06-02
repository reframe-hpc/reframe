# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.launchers import LauncherWrapper


class DdtCheck(rfm.RegressionTest):
    def __init__(self, lang, extension):
        super().__init__()
        self.name = 'DdtCheck_' + lang.replace('+', 'p')
        self.descr = 'DDT check for %s' % lang
        self.lang = lang
        self.extension = extension
        self.build_system = 'Make'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.executable = './jacobi'
        self.sourcesdir = os.path.join('src', lang)
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['tiger']:
            self.modules = ['forge']
        else:
            self.modules = ['ddt']

        self.prgenv_flags = {
            'PrgEnv-gnu': ['-g', '-O2', '-fopenmp'],
        }
        if self.current_system.name in ['arolla', 'kesch', 'tsa']:
            self.exclusive_access = True

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 4
        self.num_iterations = 5
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
        }
        self.instrumented_linenum = {
            'F90': 90,
            'C': 91,
            'C++': 94,
            'Cuda': 94
        }
        self.maintainers = ['MKr', 'JG']
        self.tags = {'production', 'craype'}
        self.postrun_cmds = ['ddt -V ; which ddt ;']
        self.keep_files = ['ddtreport.txt']

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt',
                                            self.ddt_options)


@rfm.parameterized_test(['F90', 'F90'], ['C++', 'cc'])
class DdtCpuCheck(DdtCheck):
    def __init__(self, lang, extension):
        super().__init__(lang, extension)
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:cn', 'tiger:gpu', 'arolla:cn', 'tsa:cn']
        if (self.current_system.name in ['arolla', 'kesch', 'tsa'] and
            self.lang == 'C'):
            self.build_system.ldflags = ['-lm']

        residual_pattern = '_jacobi.%s:%d,residual'
        self.ddt_options = [
            '--offline', '--output=ddtreport.txt', '--trace-at',
            residual_pattern % (
                self.extension, self.instrumented_linenum[self.lang])
        ]
        self.sanity_patterns = sn.all([
            sn.assert_found('MPI implementation', 'ddtreport.txt'),
            sn.assert_found(r'Debugging\s*:\s*srun\s+%s' % self.executable,
                            'ddtreport.txt'),
            sn.assert_reference(sn.extractsingle(
                r'^tracepoint\s+.*\s+residual:\s+(?P<result>\S+)',
                'ddtreport.txt', 'result', float), 2.572e-6, -1e-1, 1.0e-1),
            sn.assert_found(r'Every process in your program has terminated\.',
                            'ddtreport.txt')
        ])


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['Cuda', 'cu'])
class DdtGpuCheck(DdtCheck):
    def __init__(self, lang, extension):
        super().__init__(lang, extension)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn']
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.system_modules = {
            'arolla': ['cuda/10.1.243'],
            'daint': ['craype-accel-nvidia60'],
            'dom': ['craype-accel-nvidia60'],
            'kesch': ['cudatoolkit/8.0.61'],
            'tiger': ['craype-accel-nvidia60'],
            'tsa': ['cuda/10.1.243']
        }
        sysname = self.current_system.name
        self.modules += self.system_modules.get(sysname, [])

        # as long as cuda/9 will not be the default, we will need:
        if sysname in {'daint', 'kesch'}:
            self.variables = {'ALLINEA_FORCE_CUDA_VERSION': '8.0'}
        elif sysname in {'arolla', 'tsa'}:
            self.variables = {'ALLINEA_FORCE_CUDA_VERSION': '10.1'}

        self.ddt_options = [
            '--offline --output=ddtreport.txt ',
            '--break-at _jacobi-cuda-kernel.cu:59 --evaluate *residue_d ',
            '--trace-at _jacobi-cuda-kernel.cu:111,residue'
        ]
        self.build_system.cppflags = ['-DUSE_MPI', '-D_CSCS_ITMAX=5']
        if self.current_system.name == 'kesch':
            arch = 'sm_37'
            self.build_system.ldflags = ['-lm', '-lcudart']
        elif self.current_system.name in ['arolla', 'tsa']:
            arch = 'sm_70'
            self.build_system.ldflags = ['-lstdc++', '-lm',
                                         '-L$EBROOTCUDA/lib64',
                                         '-lcudart']
        else:
            arch = 'sm_60'
            self.build_system.ldflags = ['-lstdc++']

        self.build_system.options = ['NVCCFLAGS="-g -arch=%s"' % arch]
        self.sanity_patterns = sn.all([
            sn.assert_found('MPI implementation', 'ddtreport.txt'),
            sn.assert_found('Evaluate', 'ddtreport.txt'),
            sn.assert_found(r'\*residue_d:', 'ddtreport.txt'),
            sn.assert_found(r'Debugging\s*:\s*srun\s+%s' % self.executable,
                            'ddtreport.txt'),
            sn.assert_lt(sn.abs(sn.extractsingle(
                r'^tracepoint\s+.*\s+residue:\s+(?P<result>\S+)',
                'ddtreport.txt', 'result', float) - 0.25), 1e-5),
            sn.assert_found(r'Every process in your program has terminated\.',
                            'ddtreport.txt')
        ])
