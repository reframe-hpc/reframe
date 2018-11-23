import itertools
import os

import reframe as rfm
import reframe.utility.sanity as sn


class GromacsBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, output_file):
        super().__init__()

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.executable = 'gmx_mpi'

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Gromacs')
        self.keep_files = [output_file]

        energy = sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                  r'\s+Conserved En\.\s+Temperature\n'
                                  r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                  r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                  output_file, 'energy', float, item=-1)
        energy_reference = -3270799.9

        self.sanity_patterns = sn.all([
            sn.assert_found('Finished mdrun', output_file),
            sn.assert_reference(energy, energy_reference, -0.001, 0.001)
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
    def __init__(self, variant):
        super().__init__('md.log')

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'GROMACS GPU check'
        self.name = 'gromacs_gpu_%s_check' % variant
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


@rfm.simple_test
class GromacsGPUMaintCheck(GromacsGPUCheck):
    def __init__(self):
        super().__init__('maint')
        self.tags |= {'maintenance'}
        self.reference = {
            'dom:gpu': {
                'perf': (29.3, -0.05, None)
            },
            'daint:gpu': {
                'perf': (42.0, -0.10, None)
            },
        }


@rfm.simple_test
class GromacsGPUProdCheck(GromacsGPUCheck):
    def __init__(self):
        super().__init__('prod')
        self.tags |= {'production'}
        self.reference = {
            'dom:gpu': {
                'perf': (29.3, -0.05, None)
            },
            'daint:gpu': {
                'perf': (42.0, -0.20, None)
            },
        }


class GromacsCPUCheck(GromacsBaseCheck):
    def __init__(self, variant):
        super().__init__('md.log')

        self.valid_systems = ['daint:mc', 'dom:mc']
        self.descr = 'GROMACS CPU check'
        self.name = 'gromacs_cpu_%s_check' % variant
        self.executable_opts = ('mdrun -dlb yes -ntomp 1 -npme -1 '
                                '-nb cpu -s herflat.tpr ').split()

        if self.current_system.name == 'dom':
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36


@rfm.simple_test
class GromacsCPUProdCheck(GromacsCPUCheck):
    def __init__(self):
        super().__init__('prod')
        self.tags |= {'production'}
        self.reference = {
            'dom:mc': {
                'perf': (42.7, -0.05, None)
            },
            'daint:mc': {
                'perf': (70.4, -0.20, None)
            },
        }


# FIXME: This test is obsolete; it is kept only for reference.
@rfm.parameterized_test([1], [2], [4], [6], [8])
class GromacsCPUMonchAcceptance(GromacsBaseCheck):
    def __init__(self, num_nodes):
        super().__init__('md.log')

        self.valid_systems = ['monch:compute']
        self.descr = 'GROMACS %d-node CPU check on monch' % num_nodes
        self.name = 'gromacs_cpu_monch_%d_node_check' % num_nodes
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
