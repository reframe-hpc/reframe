import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['C++'], ['F90'])
class Intel_Inspector(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        self.name = 'Intel_Inspector_%s' % lang.replace('+', 'p')
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-cray': ['-O2', '-g', '-homp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp']
        }
        self.sourcesdir = os.path.join('src', lang)
        self.executable = 'inspxe-cl -collect mi1 ./jacobi'
        self.build_system = 'Make'
        if lang == 'F90':
            self.build_system.max_concurrency = 1
        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_iterations = 10
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.version_rpt = 'Intel_Inspector_version.rpt'
        self.summary_rpt = 'Intel_Inspector_summary.rpt'
        self.problems_rpt = 'Intel_Inspector_problems.rpt'
        self.observations_rpt = 'Intel_Inspector_observations.rpt'
        self.pre_run = [
            'source $INTEL_PATH/../inspector/inspxe-vars.sh',
            'inspxe-cl -h collect'
        ]
        self.post_run = [
            'inspxe-cl -V &> %s' % self.version_rpt,
            'inspxe-cl -report=summary &> %s' % self.summary_rpt,
            'inspxe-cl -report=problems &> %s' % self.problems_rpt,
            'inspxe-cl -report=observations &> %s' % self.observations_rpt,
        ]
        self.maintainers = ['JG']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        regexversion = (r'^Intel\(R\)\sInspector\s\d+\sUpdate\s\d+\s\(build',
                        r'\s(?P<toolsversion>\d+)')
        if self.current_system.name == 'dom':
            toolsversion = '579146'
        if self.current_system.name == 'daint':
            toolsversion = '551023'
        self.sanity_patterns = sn.all([
            # check the job:
            sn.assert_found('SUCCESS', self.stdout),
            # check the version:
            sn.assert_eq(sn.extractsingle(regexversion, self.version_rpt,
                         'toolsversion'), toolsversion),
            # check the reports:
            sn.assert_found(r'1 Memory leak problem\(s\) detected',
                            self.summary_rpt),
            sn.assert_found(r'1 Memory not deallocated problem\(s\) detected',
                            self.summary_rpt),
            sn.assert_found(r'_main.\w+\(\d+\): Warning X\d+: P\d: '
                            r'Memory not deallocated:',
                            self.observations_rpt),
            sn.assert_found(r'_main.\w+\(\d+\): Warning X\d+:',
                            self.problems_rpt),
        ])
