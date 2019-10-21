import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['Cuda'], ['C++'], ['F90'])
class PerftoolsCheck(rfm.RegressionTest):
    def __init__(self, lang):
        super().__init__()
        self.name = 'jacobi_perftools_%s' % lang.replace('+', 'p')
        self.descr = '%s check' % lang
        if lang != 'Cuda':
            self.valid_systems = ['daint:gpu', 'dom:gpu',
                                  'daint:mc', 'dom:mc']
        else:
            self.valid_systems = ['daint:gpu', 'dom:gpu']

        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-cray_classic',
                                    'PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi']
        if lang == 'Cpp':
            self.sourcesdir = os.path.join('src', 'C++')
        else:
            self.sourcesdir = os.path.join('src', lang)

        self.modules = ['perftools-lite']
        if lang == 'Cuda':
            self.modules += ['craype-accel-nvidia60']

        self.build_system = 'Make'
        # NOTE: Restrict concurrency to allow creation of Fortran modules
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.prgenv_flags = {
            'PrgEnv-cray': ['-O2', '-g', '-h nomessage=3140',
                            '-homp' if lang == 'F90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-O2', '-g', '-h nomessage=3140',
                                    '-homp'],
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp']
        }

        self.num_iterations = 200
        if lang == 'Cuda':
            self.build_system.options = [
                'NVCCFLAGS="-arch=sm_60"',
                'DDTFLAGS="-DUSE_MPI -D_CSCS_ITMAX=%s"' % self.num_iterations,
                'LIB=-lstdc++']

        self.executable = 'jacobi'
        # NOTE: Reduce time limit because for PrgEnv-pgi even if the output
        # is correct, the batch job uses all the time.
        self.time_limit = (0, 5, 0)

        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        if lang == 'Cuda':
            self.num_gpus_per_node = 1
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 1

        self.variables = {
            'ITERATIONS': str(self.num_iterations),
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic'
        }
        if self.num_tasks == 1:
            # will be fixed in perftools/7.1
            self.variables['PAT_RT_REPORT_METHOD'] = 'pe'

        self.sanity_patterns = sn.assert_found('Table 1:  Profile by Function',
                                               self.stdout)
        self.maintainers = ['MK', 'JG']
        self.tags = {'production', 'craype'}

    def setup(self, environ, partition, **job_opts):
        super().setup(environ, partition, **job_opts)
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = flags
        self.build_system.cxxflags = flags
        self.build_system.fflags = flags
