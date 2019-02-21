import os

import reframe as rfm
import reframe.utility.sanity as sn


class FieldextraCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.maintainers = ['Mkr']

        self.valid_systems = ['kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu-nompi']
        self.executable = 'cookbook/run.bash'

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 24
        self.num_task_per_core = 1
        self.use_multithreading = True
        self.strict_check = False


@rfm.parameterized_test(['opt'], ['opt_omp'])
class FieldextraAccuracy(FieldextraCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = (
            'Fieldextra validation test (accuracy); MCH'
        )
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Fieldextra', 'cookbook_tests')
        self.readonly_files = [
            'cookbook/support/input',
            'reference_cookbook'
        ]
        self.pre_run = [
            'ln -s ${EBROOTFIELDEXTRA}/bin bin',
            'ln -s ${EBROOTFIELDEXTRA}/tools tools',
            'ln -s ${EBROOTFIELDEXTRA}/resources resources',
        ]

        if variant == 'opt':
            self.modules = ['fieldextra/12.7.5-gmvolf-17.02']
            self.executable_opts = ['-c gnu -m opt']
        else:
            self.modules = ['fieldextra/12.7.5-gmvolf-17.02-openmp']
            self.executable_opts = ['-c gnu -m opt_omp']
            self.variables = {
                'OMP_STACKSIZE': '500M',
                'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
            }

        self.sanity_patterns = sn.assert_found(r'All tests successful',
                                               self.stdout)


@rfm.simple_test
class FieldextraPerformance(FieldextraCheck):
    def __init__(self):
        super().__init__()
        self.descr = (
            'Fieldextra validation test (performance); MCH'
        )
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Fieldextra', 'performance')
        self.modules = ['fieldextra/12.7.5-gmvolf-17.02-openmp']
        self.executable = 'fieldextra_gnu_opt_omp'
        self.executable_opts = ['nl.TC']
        self.readonly_files = [
            '18112900_204',
        ]
        self.pre_run = [
            'source create_nl_6h.template'
        ]
        self.variables = {
             'OMP_STACKSIZE': '500M',
             'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }

        self.sanity_patterns = sn.assert_found(
            r'%INFO fieldextra: Program successfully completed', self.stdout
        )
        self.perf_patterns = {
            'perf': sn.extractsingle(r'WALL CLOCK\s*SPEEDUP\D*(?P<perf>\S+)',
                                     'fieldextra.diagnostic', 'perf', float)
        }
        self.reference = {
            'kesch': {
                'perf': (350., None, 0.20)
            }
        }
