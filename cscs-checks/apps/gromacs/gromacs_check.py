# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
from hpctestlib.sciapps.gromacs.checks import gromacs_check


@rfm.simple_test
class cscs_gromacs_check(gromacs_check):
    modules = ['GROMACS']
    maintainers = ['VH', 'VK']
    use_multithreading = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    executable_opts += ['-dlb yes', '-ntomp 1', '-npme -1']
    valid_prog_environs = ['builtin']

    # CSCS-specific parameterization
    num_nodes = parameter([6, 16])
    mode = parameter(['maintenance', 'production'])
    allref = {
        6: {
            'sm_60': {
                'HECBioSim/Crambin': (0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (0, None, None, 'ns/day'),
            },
            'broadwell': {
                'HECBioSim/Crambin': (0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (0, None, None, 'ns/day'),
            },
            'zen2': {
                'HECBioSim/Crambin': (0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (0, None, None, 'ns/day'),
            },
        },
        16: {
            'sm_60': {
                'HECBioSim/Crambin': (0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (0, None, None, 'ns/day'),
            },
            'broadwell': {
                'HECBioSim/Crambin': (0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (0, None, None, 'ns/day'),
            },
            'zen2': {
                'HECBioSim/Crambin': (0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (0, None, None, 'ns/day'),
            },
        }
    }

    @run_after('init')
    def setup_filtering_criteria(self):
        # Update test's description
        self.descr += f' ({self.num_nodes} node(s), {self.mode!r} mode)'

        # Setup system filtering
        valid_systems = {
            'cpu': {
                6:  ['daint:mc', 'dom:mc', 'eiger:mc', 'pilatus:mc'],
                16: ['daint:mc', 'eiger:mc']
            },
            'gpu': {
                6:  ['daint:gpu', 'dom:gpu', 'eiger:gpu', 'pilatus:gpu'],
                16: ['daint:gpu', 'eiger:gpu']
            }
        }
        try:
            self.valid_systems = valid_systems[self.nb_impl][self.num_nodes]
        except KeyError:
            self.valid_systems = []

        # Maintenance mode is not valid for the cpu run
        if self.nb_impl == 'cpu' and self.mode == 'maintenance':
            self.valid_systems = []

        # Setup prog env. filtering
        if self.current_system.name in ('eiger', 'pilatus'):
            self.valid_prog_environs = ['cpeGNU']

        self.tags |= {self.mode}

    @run_before('run')
    def setup_run(self):
        # self.skip_if_no_procinfo()
        # Setup GPU run
        if self.nb_impl == 'gpu':
            self.num_gpus_per_node = 1
            self.variables = {'CRAY_CUDA_MPS': '1'}

        proc = self.current_partition.processor

        # Choose arch; we set explicitly the GPU arch, since there is no
        # auto-detection
        arch = proc.arch
        if self.current_partition.fullname in ('daint:gpu', 'dom:gpu'):
            arch = 'sm_60'

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][arch][self.bench_name]
            }
        }

        # Setup parallel run
        self.num_tasks_per_node = proc.num_cores
        self.num_tasks = self.num_nodes * self.num_tasks_per_node


# FIXME: Remove the following references
REFERENCE_GPU_PERFORMANCE = {
    'large': {
        'daint:gpu': {
            'maint': (63.0, -0.10, None, 'ns/day'),
            'prod': (63.0, -0.20, None, 'ns/day'),
        },
    },
    'small': {
        'daint:gpu': {
            'prod': (35.0, -0.10, None, 'ns/day'),
        },
        'dom:gpu': {
            'prod': (37.0, -0.05, None, 'ns/day'),
        },
    }
}

REFERENCE_CPU_PERFORMANCE = {
    'large': {
        'daint:mc': {
            'prod': (68.0, -0.20, None, 'ns/day'),
        },
        'eiger:mc': {
            'prod': (146.00, -0.20, None, 'ns/day'),
        },
        'pilatus:mc': {
            'prod': (146.00, -0.20, None, 'ns/day'),
        },
    },
    'small': {
        'daint:mc': {
            'prod': (38.8, -0.10, None, 'ns/day'),
        },
        'dom:mc': {
            'prod': (40.0, -0.05, None, 'ns/day'),
        },
        'eiger:mc': {
            'prod': (103.00, -0.10, None, 'ns/day'),
        },
        'dom:mc': {
            'prod': (103.00, -0.10, None, 'ns/day'),
        },
    }
}
