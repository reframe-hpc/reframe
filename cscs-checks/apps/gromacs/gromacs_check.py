import itertools
import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class GromacsBaseCheck(RunOnlyRegressionTest):
    def __init__(self, name, output_file, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.executable = 'gmx_mpi'

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Gromacs')
        self.keep_files = [output_file]

        energy = sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                  r'\s+Temperature\s+Pressure \(bar\)\n'
                                  r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                  r'\s+Constr\. rmsd',
                                  output_file, 'energy', float, item=-1)
        energy_reference = -3270799.9
        energy_diff = sn.abs(energy - energy_reference)

        self.sanity_patterns = sn.all([
            sn.assert_found('Finished mdrun', output_file),
            sn.assert_lt(energy_diff, 1560.1)
        ])

        self.perf_patterns = {
            'perf': sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                     output_file, 'perf', float)
        }

        self.modules = ['GROMACS']
        self.maintainers = ['VH']
        self.strict_check = False
        self.use_multithreading = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class GromacsGPUCheck(GromacsBaseCheck):
    def __init__(self, variant, **kwargs):
        super().__init__('gromacs_gpu_%s_check' % variant,
                         'md.log', **kwargs)

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'GROMACS GPU check'
        self.executable_opts = ('mdrun -dlb yes -ntomp 1 -npme 0 '
                                '-s herflat.tpr ').split()
        self.variables = {'CRAY_CUDA_MPS': '1'}
        self.tags = {'scs'}
        self.num_gpus_per_node = 1

        if self.current_system.name == 'dom':
            self.num_tasks = 72
            self.num_tasks_per_node = 12
        else:
            self.num_tasks = 192
            self.num_tasks_per_node = 12


class GromacsGPUMaintCheck(GromacsGPUCheck):
    def __init__(self, **kwargs):
        super().__init__('maint', **kwargs)
        self.tags |= {'maintenance'}
        self.reference = {
            'dom:gpu': {
                'perf': (29.3, -0.05, None)
            },
            'daint:gpu': {
                'perf': (60.2, -0.10, None)
            },
        }


class GromacsGPUProdCheck(GromacsGPUCheck):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'dom:gpu': {
                'perf': (32.0, -0.05, None)
            },
            'daint:gpu': {
                'perf': (47.5, -0.40, None)
            },
        }


class GromacsCPUCheck(GromacsBaseCheck):
    def __init__(self, variant, **kwargs):
        super().__init__('gromacs_cpu_%s_check' % variant,
                         'md.log', **kwargs)

        self.valid_systems = ['daint:mc', 'dom:mc']
        self.descr = 'GROMACS CPU check'
        self.executable_opts = ('mdrun -dlb yes -ntomp 1 -npme -1 '
                                '-nb cpu -s herflat.tpr ').split()

        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36


class GromacsCPUProdCheck(GromacsCPUCheck):
    def __init__(self, **kwargs):
        super().__init__('prod', **kwargs)
        self.tags |= {'production'}
        self.reference = {
            'dom:mc': {
                'perf': (38.0, -0.05, None)
            },
            'daint:mc': {
                'perf': (73.0, -0.50, None)
            },
        }


class GromacsCPUMonchAcceptance(GromacsBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        nodes_label = 'node' if num_nodes == 1 else 'nodes'
        super().__init__('gromacs_cpu_monch_%d_%s_check'
                         % (num_nodes, nodes_label), 'md.log', **kwargs)

        self.valid_systems = ['monch:compute']
        self.descr = 'GROMACS CPU check on %d %s on monch' % (num_nodes,
                                                              nodes_label)

        self.executable_opts = ('mdrun -dlb yes -ntomp 1 -npme -1 '
                                '-nsteps 5000 -nb cpu -s herflat.tpr ').split()

        self.tags = {'monch_acceptance'}
        self.num_tasks_per_node = 20
        self.num_tasks = num_nodes * self.num_tasks_per_node

        reference_by_nodes = {1: 2.6, 2: 5.1, 4: 11.1, 6: 15.8, 8: 20.6}

        self.reference = {
            'monch:compute': {
                'perf': (reference_by_nodes[num_nodes], -0.15, None)
            }
        }


def _get_checks(**kwargs):
    return list(itertools.chain(
        [GromacsGPUMaintCheck(**kwargs),
         GromacsGPUProdCheck(**kwargs),
         GromacsCPUProdCheck(**kwargs)],
        [GromacsCPUMonchAcceptance(n, **kwargs) for n in [1, 2, 4, 6, 8]]))
