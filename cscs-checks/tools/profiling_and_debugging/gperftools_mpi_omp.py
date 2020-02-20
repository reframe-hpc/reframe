# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['Cpp'], ['F90'])
class GperftoolsMpiCheck(rfm.RegressionTest):
    '''This test checks gperftools:
    https://gperftools.github.io/gperftools/cpuprofile.html
    '''

    def __init__(self, lang):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-cray': ['-O2', '-g',
                            '-homp' if lang == 'F90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-O2', '-g', '-homp'],
            'PrgEnv-gnu': ['-g', '-fopenmp', '-O2'],
            'PrgEnv-intel': ['-g', '-qopenmp', '-O2'],
            'PrgEnv-pgi': ['-g', '-mp', '-O2']
        }
        # external pprof is needed to avoid "stack trace depth >= 2**32" errors
        self.modules = ['gperftools', 'graphviz', 'pprof']
        self.build_system = 'Make'
        self.iterations = 500
        self.build_system.cppflags = [
            '-DUSE_MPI',
            '-D_CSCS_ITMAX=%s' % self.iterations,
        ]
        if lang == 'Cpp':
            self.sourcesdir = os.path.join('src', 'C++')
        else:
            self.sourcesdir = os.path.join('src', lang)

        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.num_tasks = 96
        self.num_tasks_per_node = 24
        self.num_cpus_per_task = 1
        self.num_tasks_per_core = 2
        self.num_iterations = self.iterations
        self.split_file = '0.sh'
        self.executable = self.split_file
        self.exe = './jacobi'
        self.rpt_file = 'gperftools.rpt'
        self.rpt_file_txt = '%s.txt' % self.rpt_file
        self.rpt_file_pdf = '%s.pdf' % self.rpt_file
        self.rpt_file_doc = '%s.doc' % self.rpt_file
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.pre_run = [
            'echo \'#!/bin/bash\' &> %s' % self.split_file,
            'echo \'CPUPROFILE=`hostname`.$SLURM_PROCID\' %s >> %s' %
            (self.exe, self.split_file),
            'chmod u+x %s' % (self.split_file),
        ]
        self.post_run = [
            '$EBROOTPPROF/bin/pprof --unit=ms --text --lines %s %s &> %s' %
            (self.exe, '*.0', self.rpt_file_txt),
            '$EBROOTPPROF/bin/pprof --pdf %s %s &> %s' %
            (self.exe, '*.0', self.rpt_file_pdf),
            'file %s &> %s' % (self.rpt_file_pdf, self.rpt_file_doc)
        ]
        self.sanity_patterns = sn.all([
            # check job status:
            sn.assert_found('SUCCESS', self.stdout),
            # check txt report:
            sn.assert_found(
                r'^\s+\d+ms\s+\d+.\d+%.*_jacobi.\w+:\d+', self.rpt_file_txt),
            # check pdf report:
            sn.assert_found('PDF document', self.rpt_file_doc),
        ])
        self.perf_patterns = { 'jacobi_elapsed%': self.report_flat_pctg, }
        self.maintainers = ['JG']
        self.tags = {'performance-tools'}

    def setup(self, environ, partition, **job_opts):
        super().setup(environ, partition, **job_opts)
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = flags
        self.build_system.cxxflags = flags
        self.build_system.fflags = flags
        self.build_system.ldflags = flags + ['`pkg-config --libs libprofiler`']

# {{{
    @property
    @sn.sanity_function
    def report_flat_pctg(self):
        '''
        A typical report looks like:
        flat  flat%   sum%        cum   cum%
        50ms 15.62% 15.62%       50ms 15.62%  GNII_DlaProgress ??:?
        30ms  9.38% 50.00%       30ms  9.38
          L__Z6JacobiR10JacobiData_67__par_region1_2_2 GperftoolsMpiCheck_Cpp/
          _jacobi.cc:82
        This functions extracts flat% of the first jacobi function call found.
        '''
        regex = r'^\s+\d+ms\s+(?P<pctg>\d+.\d+)%.*_jacobi.\w+:\d+'
        result = sn.extractsingle(regex, self.rpt_file_txt, 'pctg', float)
        return result
# }}}
