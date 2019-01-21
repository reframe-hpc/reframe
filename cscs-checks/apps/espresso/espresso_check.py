import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('<=2.14')
class EspressoBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, variant):
        super().__init__()

        self.descr = 'Quantum Espresso check (%s)' % variant
        self.maintainers = ['AK', 'LM']

        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Espresso')
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['QuantumESPRESSO']
        self.executable = 'pw.x'
        self.executable_opts = '-in ausurf.in'.split()
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
            sn.assert_reference(energy, -11427.09017162, -1e-10, 1e-10)])
        self.perf_patterns = {
            'sec': sn.extractsingle(r'electrons    :\s+(?P<sec>\S+)s CPU ',
                                    self.stdout, 'sec', float)
        }


@rfm.parameterized_test(['cpu'], ['gpu'])
class EspressoProdCheck(EspressoBaseCheck):
    def __init__(self, variant):
        super().__init__(variant)

        if variant == 'cpu':
            self.tags = {'scs', 'production'}
            self.valid_systems = ['daint:mc', 'dom:mc']
            self.reference = {
                'dom:mc': {
                    'sec': (159.0, None, 0.05),
                },
                'daint:mc': {
                    'sec': (157.0, None, 0.40)
                },
            }
        else:
            self.valid_systems = ['daint:gpu', 'dom:gpu']
            self.executable = 'pw-gpu.x'
            self.use_multithreading = True
            if self.current_system.name == 'dom':
                self.num_tasks = 72
                self.num_tasks_per_node = 12
            else:
                self.num_tasks = 192
                self.num_tasks_per_node = 12
            self.reference = {
                'dom:gpu': {
                    # FIXME: Update this value as soon as GPU version is working
                    'sec': (0.097, None, 0.15),
                },
                'daint:gpu': {
                    # FIXME: Update this value as soon as GPU version is working
                    'sec': (0.097, None, 0.15),
                },
            }

