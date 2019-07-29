import os

import reframe as rfm
import reframe.utility.sanity as sn


# @rfm.parameterized_test(['C++'], ['F90'])
@rfm.parameterized_test(['C++'])
class IntelInspectorTest(rfm.RegressionTest):
    '''This test checks Intel Inspector:
    https://software.intel.com/en-us/inspector
    '''
    def __init__(self, lang):
        super().__init__()
        self.name = 'Intel_Inspector_%s' % lang.replace('+', 'p')
        self.descr = self.name
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['inspector']
        self.sourcesdir = os.path.join('src', lang)
        self.build_system = 'Make'
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.executable = 'inspxe-cl'
        self.target_executable = './jacobi'
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-g', '-O2', '-fopenmp'],
            'PrgEnv-cray': ['-g', '-O2', '-homp'],
            'PrgEnv-intel': ['-g', '-O2', '-qopenmp'],
            'PrgEnv-pgi': ['-g', '-O2', '-mp']
        }
        self.executable_opts = ['-collect mi1 %s' % self.target_executable]
        self.exclusive = True
        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        num_iterations = 10
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.version_rpt = 'version.rpt'
        self.problems_rpt = 'problems.rpt'
        self.summary_rpt = 'summary.rpt'
        self.observations_rpt = 'observations.rpt'
        self.pre_run = [
            'mv %s %s' % (self.executable, self.target_executable),
            '%s --version &> %s' % (self.executable, self.version_rpt),
        ]
        self.post_run = [
            '%s -V &> %s' % (self.executable, self.version_rpt),
            '%s -report=summary &> %s' % (self.executable, self.summary_rpt),
            '%s -report=problems &> %s' % (self.executable, self.problems_rpt),
            '%s -report=observations &> %s' %
            (self.executable, self.observations_rpt),
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
        regexversion = (r'^Intel\(R\)\sInspector\s\d+\sUpdate\s\d+\s\(build'
                        r'\s(?P<toolsversion>\d+)')
        system_default_toolversion = {
            'daint': '551023',  # 2018 Update 2
            'dom': '597413',    # 2019 Update 4
        }
        toolsversion = system_default_toolversion[self.current_system.name]
        self.sanity_patterns = sn.all([
            # check the job:
            sn.assert_found('SUCCESS', self.stdout),
            # check the tool's version:
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
