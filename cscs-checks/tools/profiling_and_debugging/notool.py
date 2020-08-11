# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.fields import ScopedDict


@rfm.parameterized_test(['C++'], ['F90'])
class JacobiNoToolHybrid(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        self.descr = 'Jacobi (without tool) %s check' % lang
        self.name = '%s_%s' % (type(self).__name__, lang.replace('+', 'p'))
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-cray_classic',
                                    'PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-cray': ['-O2', '-g',
                            '-homp' if lang == 'F90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-O2', '-g', '-homp'],
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp']
        }
        self.sourcesdir = os.path.join('src', lang)
        self.build_system = 'Make'
        self.executable = './jacobi'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
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
            'CRAYPE_LINK_TYPE': 'dynamic'
        }
        # OpenMP support varies between compilers:
        self.openmp_versions = ScopedDict({
            'PrgEnv-cray': {'version': 201511},
            'PrgEnv-cray_classic': {'version': 201511},
            'PrgEnv-gnu': {'version': 201511},
            'PrgEnv-intel': {'version': 201611},
            'PrgEnv-pgi': {'version': 201307},
        })
        self.lang = lang
        # The scopedict (above) is better than this:
        # if (self.lang == 'C++' and
        #    self.current_environ.name == 'PrgEnv-pgi'):
        #    self.omp_versions['PrgEnv-pgi'] = '200805'
        self.maintainers = ['JG', 'MKr']
        self.tags = {'production'}
        url = 'http://github.com/eth-cscs/hpctools'
        readme_str = (r'More debug and performance tools ReFrame checks are'
                      r' available at %s' % url)
        self.postrun_cmds = ['echo "%s"' % readme_str]
        if self.current_system.name in {'dom', 'daint'}:
            # get general info about the environment:
            self.postrun_cmds += ['module list -t']
        self.perf_patterns = {
            'elapsed_time': sn.extractsingle(r'Elapsed Time\s*:\s+(\S+)',
                                             self.stdout, 1, float)
        }
        self.reference = {
            '*': {
                'elapsed_time': (0, None, None, 's')
            }
        }
        if lang == 'C++':
            self.reference_lang = (0.38, -0.6, None, 's')
        elif lang == 'F90':
            self.reference_lang = (0.17, -0.6, None, 's')

    @rfm.run_before('compile')
    def set_flags(self):
        envname = self.current_environ.name
        # if generic, falls back to -g:
        prgenv_flags = self.prgenv_flags.get(envname, ['-g'])
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.build_system.ldflags = ['-lm']

    @rfm.run_before('sanity')
    def set_sanity(self):
        found_version = sn.extractsingle(r'OpenMP-\s*(\d+)', self.stdout, 1,
                                         int)
        envname = self.current_environ.name
        ompversion_key = '%s:%s:version' % (envname, self.lang)
        self.sanity_patterns = sn.all([
            sn.assert_eq(found_version, self.openmp_versions[ompversion_key]),
            sn.assert_found('SUCCESS', self.stdout),
        ])

    @rfm.run_before('sanity')
    def set_reference(self):
        if self.current_system.name in {'dom', 'daint'}:
            self.reference['*:elapsed_time'] = self.reference_lang
