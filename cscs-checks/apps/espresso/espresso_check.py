import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16')
@rfm.parameterized_test(['small'], ['large'])
class QECheck(rfm.RunOnlyRegressionTest):
    def __init__(self, scale):
        super().__init__()
        self.descr = 'Quantum Espresso CPU check'
        self.maintainers = ['LM', 'CB']
        self.tags = {'scs', 'production', 'external-resources'}
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Espresso')

        self.valid_systems = ['daint:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['QuantumESPRESSO']
        self.executable = 'pw.x'
        self.executable_opts = ['-in', 'ausurf.in']
        if scale == 'small':
            self.valid_systems += ['dom:mc']
            self.num_tasks = 216
            self.num_tasks_per_node = 36
            self.reference = {
                'dom:mc': {
                    'time': (159.0, None, 0.05, 's'),
                },
                'daint:mc': {
                    'time': (151.6, None, 0.05, 's')
                },
            }
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36
            self.reference = {
                'daint:mc': {
                    'time': (157.0, None, 0.40, 's')
                },
            }

        self.use_multithreading = True
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

        self.strict_check = False
        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_reference(energy, -11427.09017162, -1e-10, 1e-10)
        ])
        self.perf_patterns = {
            'time': sn.extractsingle(r'electrons    :\s+(?P<sec>\S+)s CPU ',
                                     self.stdout, 'sec', float)
        }
