from datetime import datetime

import reframe as rfm
import reframe.utility.sanity as sn


class HelloWorldBaseTest(rfm.RegressionTest):
    def __init__(self, name, lang, linkage):
        super().__init__()
        if self.current_system.name in ['dom', 'daint']:
            self.modules += ['gcc/6.1.0']

        self.variables = {'CRAYPE_LINK_TYPE': linkage}
        self.prgenv_flags = {}
        self.lang_names = {
            'c': 'C',
            'cpp': 'C++',
            'f90': 'Fortran 90'
        }
        self.descr = self.lang_names[lang] + ' Hello World'
        self.sourcepath = 'hello_world'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn', 'leone:normal',
                              'monch:compute']

        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']

        # Removing static compilation from kesch
        if (self.current_system.name in ['kesch', 'leone', 'monch'] and
            linkage == 'static'):
            self.valid_prog_environs = []

        self.compilation_time_seconds = None

        self.maintainers = ['CB', 'VK']
        self.tags = {'production', 'monch_acceptance'}

    def setup(self, partition, environ, **job_opts):

        result = sn.findall(r'Hello World from thread \s*(\d+) out '
                            r'of \s*(\d+) from process \s*(\d+) out of '
                            r'\s*(\d+)', self.stdout)

        self.sanity_patterns = sn.all(
            sn.chain([sn.assert_eq(sn.count(result), self.num_tasks *
                                   self.num_cpus_per_task)],
                     sn.map(
                         lambda x: sn.assert_lt(int(x.group(1)),
                                                int(x.group(2))),
                         result),
                     sn.map(
                         lambda x: sn.assert_lt(int(x.group(3)),
                                                int(x.group(4))),
                         result),
                     sn.map(
                         lambda x: sn.assert_lt(int(x.group(1)),
                                                self.num_cpus_per_task),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(int(x.group(2)),
                                                self.num_cpus_per_task),
                         result),
                     sn.map(
                         lambda x: sn.assert_lt(int(x.group(3)),
                                                self.num_tasks),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(int(x.group(4)),
                                                self.num_tasks),
                         result),
                     )
        )

        self.perf_patterns = {
            'compilation_time': sn.getattr(self, 'compilation_time_seconds')
        }
        self.reference = {
            '*': {
                'compilation_time': (60, None, 0.1)
            }
        }
        super().setup(partition, environ, **job_opts)
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags

    def compile(self):
        self.compilation_time_seconds = datetime.now()
        super().compile()
        self.compilation_time_seconds = (
            datetime.now() - self.compilation_time_seconds).total_seconds()


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestSerial(HelloWorldBaseTest):
    def __init__(self, lang, linkage, **kwargs):
        super().__init__('serial', lang, linkage, **kwargs)

        self.sourcepath += '_serial.' + lang
        self.descr += ' Serial ' + linkage.capitalize()
        self.prgenv_flags = {
            'PrgEnv-cray': [],
            'PrgEnv-gnu': [],
            'PrgEnv-intel': [],
            'PrgEnv-pgi': []
        }
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestOpenMP(HelloWorldBaseTest):
    def __init__(self, lang, linkage):
        super().__init__('openmp', lang, linkage)
        self.sourcepath += '_openmp.' + lang
        self.descr += ' OpenMP ' + str.capitalize(linkage)
        self.prgenv_flags = {
            'PrgEnv-cray': ['-homp'],
            'PrgEnv-gnu': ['-fopenmp'],
            'PrgEnv-intel': ['-qopenmp'],
            'PrgEnv-pgi': ['-mp']
        }
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 4

        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestMPI(HelloWorldBaseTest):
    def __init__(self, lang, linkage):
        super().__init__('mpi', lang, linkage)
        self.sourcepath += '_mpi.' + lang
        self.descr += ' MPI ' + linkage.capitalize()
        self.prgenv_flags = {
            'PrgEnv-cray': [],
            'PrgEnv-gnu': [],
            'PrgEnv-intel': [],
            'PrgEnv-pgi': []
        }

        # for the MPI test the self.num_tasks_per_node should always be one. If
        # not, the test will fail for the total number of lines in the output
        # file is different then self.num_tasks * self.num_tasks_per_node
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestMPIOpenMP(HelloWorldBaseTest):
    def __init__(self, lang, linkage):
        super().__init__('mpi_openmp', lang, linkage)
        self.sourcepath += '_mpi_openmp.' + lang
        self.descr += ' MPI + OpenMP ' + linkage.capitalize()
        self.prgenv_flags = {
            'PrgEnv-cray': ['-homp'],
            'PrgEnv-gnu': ['-fopenmp'],
            'PrgEnv-intel': ['-qopenmp'],
            'PrgEnv-pgi': ['-mp']
        }
        self.num_tasks = 6
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4

        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
