import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['Cuda'], ['C++'], ['F90'])
class PerftoolsCheck(rfm.RegressionTest):
    def __init__(self, lang):
        self.name = 'jacobi_perftools_%s' % lang.replace('+', 'p')
        self.descr = '%s check' % lang
        if lang != 'Cuda':
            self.valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc',
                                  'tiger:gpu']
        else:
            self.valid_systems = ['daint:gpu', 'dom:gpu', 'tiger:gpu']

        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-cray_classic',
                                    'PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi']
        if lang == 'Cpp':
            self.sourcesdir = os.path.join('src', 'C++')
        else:
            self.sourcesdir = os.path.join('src', lang)

        self.modules = ['perftools-lite']
        # unload xalt to avoid conflict with perftools:
        self.prebuild_cmd = ['module rm xalt ;module list -t']
        if lang == 'Cuda':
            self.modules += ['craype-accel-nvidia60']

        self.build_system = 'Make'
        if lang == 'F90':
            # NOTE: Restrict concurrency to allow creation of Fortran modules
            self.build_system.max_concurrency = 1

        self.prgenv_flags = {
            'PrgEnv-cray': ['-O2', '-g',
                            '-homp' if lang == 'F90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-O2', '-g', '-homp'],
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
        # keeping as reminder:  needed with perftools<7.1:
        # if self.num_tasks == 1:
        #     self.variables['PAT_RT_REPORT_METHOD'] = 'pe'
        self.post_run = ['grep INFO rfm_*_build.err']
        self.sanity_patterns = sn.all([
            sn.assert_found('SUCCESS', self.stdout),
            sn.assert_found(r'^INFO: creating the (\w+)-instrumented exec',
                            self.stdout),
            sn.assert_found('Table 1:  Profile by Function', self.stdout),
        ])
        self.perf_patterns = {
            'Elapsed': self.perftools_seconds,
            'High Memory per PE': self.perftools_highmem,
            'Instr per Cycle': self.perftools_ipc,
        }
        self.reference = {
            '*': {
                'Elapsed': (0, None, None, 's'),
                'High Memory per PE': (0, None, None, 'MiBytes'),
                'Instr per Cycle': (0, None, None, ''),
            }
        }
        self.maintainers = ['JG', 'MKr']
        self.tags = {'production', 'craype'}

    @rfm.run_before('compile')
    def setflags(self):
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = flags
        self.build_system.cxxflags = flags
        self.build_system.fflags = flags

    @property
    @sn.sanity_function
    def perftools_seconds(self):
        regex = r'^Avg Process Time:\s+(?P<sec>\S+) secs'
        result = sn.extractsingle(regex, self.stdout, 'sec', float)
        return result

    @property
    @sn.sanity_function
    def perftools_highmem(self):
        regex = r'High Memory:\s+\S+ MiBytes\s+(?P<MBytes>\S+) MiBytes per PE'
        result = sn.extractsingle(regex, self.stdout, 'MBytes', float)
        return result

    @property
    @sn.sanity_function
    def perftools_ipc(self):
        regex = r'^Instr per Cycle:\s+(?P<ipc>\S+)'
        result = sn.extractsingle(regex, self.stdout, 'ipc', float)
        return result
