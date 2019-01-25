import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class QECheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Quantum Espresso CPU check'
        self.maintainers = ['AK', 'LM']
        self.tags = {'scs', 'production'}
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Espresso')

        self.valid_systems = ['daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['QuantumESPRESSO']
        self.executable = 'pw.x'
        self.executable_opts = ['-in', 'ausurf.in']
        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36

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
            'sec': sn.extractsingle(r'electrons    :\s+(?P<sec>\S+)s CPU ',
                                    self.stdout, 'sec', float)
        }
        self.reference = {
            'dom:mc': {
                'sec': (159.0, None, 0.05),
            },
            'daint:mc': {
                'sec': (157.0, None, 0.40)
            },
        }

