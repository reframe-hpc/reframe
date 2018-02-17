import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RunOnlyRegressionTest


class EspressoBaseCheck(RunOnlyRegressionTest):
    def __init__(self, variant, **kwargs):
        super().__init__('quantum_espresso_%s_check' % variant,
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Quantum Espresso check (%s)' % variant
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['QuantumESPRESSO']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Espresso')
        self.executable = 'pw.x'
        self.executable_opts = '-in ausurf.in'.split()

        self.sanity_patterns = sn.assert_found(
            r'convergence has been achieved', self.stdout)
        self.perf_patterns = {
            'sec': sn.extractsingle(r'electrons    :\s+(?P<sec>\S+)s CPU ',
                                    self.stdout, 'sec', float)
        }
        self.use_multithreading = True
        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36

        self.maintainers = ['AK', 'LM']
        self.strict_check = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class EspressoCPUProdCheck(EspressoBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('cpu', **kwargs)
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.tags = {'scs', 'production'}

        self.reference = {
            'dom:mc': {
                'sec': (217.0, None, 0.15),
            },
            'daint:mc': {
                'sec': (156.6, None, 0.34)   # (151.0, None, 0.15)
            },
        }


class EspressoGPUCheck(EspressoBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('gpu', **kwargs)

        self.executable = 'pw-gpu.x'
        self.valid_systems = ['daint:gpu', 'dom:gpu']

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


def _get_checks(**kwargs):
    return [EspressoCPUProdCheck(**kwargs)]
