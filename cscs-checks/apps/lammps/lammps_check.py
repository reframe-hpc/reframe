# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class LAMMPSCheck(rfm.RunOnlyRegressionTest):
    scale = parameter(['small', 'large'])
    variant = parameter(['maint', 'prod'])
    modules = ['cray-python', 'LAMMPS']
    tags = {'scs', 'external-resources'}
    maintainers = ['LM']
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def setup_by_system(self):
        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'LAMMPS')
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @performance_function('timesteps/s')
    def perf(self):
        return sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                self.stdout, 'perf', float)

    @sanity_function
    def assert_energy_diff(self):
        energy_reference = -4.6195
        energy = sn.extractsingle(
            r'\s+500000(\s+\S+){3}\s+(?P<energy>\S+)\s+\S+\s\n',
            self.stdout, 'energy', float)
        energy_diff = sn.abs(energy - energy_reference)
        return sn.all([
            sn.assert_found(r'Total wall time:', self.stdout),
            sn.assert_lt(energy_diff, 6e-4)
        ])


@rfm.simple_test
class LAMMPSGPUCheck(LAMMPSCheck):
    valid_systems = ['daint:gpu']
    executable = 'lmp_mpi'
    executable_opts = ['-sf gpu', '-pk gpu 1', '-in in.lj.gpu']
    variables = {'CRAY_CUDA_MPS': '1'}
    num_gpus_per_node = 1
    references_by_variant = {
        'maint': {
            'small': {
                'dom:gpu': {'perf': (3457, -0.10, None, 'timesteps/s')},
                'daint:gpu': {'perf': (2524, -0.10, None, 'timesteps/s')}
            },
            'large': {
                'daint:gpu': {'perf': (3832, -0.05, None, 'timesteps/s')}
            }
        },
        'prod': {
            'small': {
                'dom:gpu': {'perf': (3132, -0.05, None, 'timesteps/s')},
                'daint:gpu': {'perf': (2400, -0.40, None, 'timesteps/s')}
            },
            'large': {
                'daint:gpu': {'perf': (3260, -0.50, None, 'timesteps/s')}
            }
        },
    }

    @run_after('init')
    def setup_by_variant(self):
        self.descr = (f'LAMMPS GPU check (version: {self.scale}, '
                      f'{self.variant})')
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 12
            self.num_tasks_per_node = 2
        else:
            self.num_tasks = 32
            self.num_tasks_per_node = 2

        self.reference = self.references_by_variant[self.variant][self.scale]
        self.tags |= {
            'maintenance' if self.variant == 'maint' else 'production'
        }


@rfm.simple_test
class LAMMPSCPUCheck(LAMMPSCheck):
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
    references_by_variant = {
        'maint': {
            'small': {
                'dom:mc': {'perf': (4394, -0.05, None, 'timesteps/s')},
                'daint:mc': {'perf': (3824, -0.10, None, 'timesteps/s')},
                'eiger:mc': {'perf': (4500, -0.10, None, 'timesteps/s')},
                'pilatus:mc': {'perf': (5000, -0.10, None, 'timesteps/s')}
            },
            'large': {
                'daint:mc': {'perf': (5310, -0.65, None, 'timesteps/s')},
                'eiger:mc': {'perf': (6500, -0.10, None, 'timesteps/s')},
                'pilatus:mc': {'perf': (7500, -0.10, None, 'timesteps/s')}
            }
        },
        'prod': {
            'small': {
                'dom:mc': {'perf': (4394, -0.05, None, 'timesteps/s')},
                'daint:mc': {'perf': (3824, -0.10, None, 'timesteps/s')},
                'eiger:mc': {'perf': (4500, -0.10, None, 'timesteps/s')},
                'pilatus:mc': {'perf': (5000, -0.10, None, 'timesteps/s')}
            },
            'large': {
                'daint:mc': {'perf': (5310, -0.65, None, 'timesteps/s')},
                'eiger:mc': {'perf': (6500, -0.10, None, 'timesteps/s')},
                'pilatus:mc': {'perf': (7500, -0.10, None, 'timesteps/s')}
            }
        }
    }

    @run_after('init')
    def setup_by_variant(self):
        self.descr = (f'LAMMPS CPU check (version: {self.scale}, '
                      f'{self.variant})')
        if self.current_system.name in ['eiger', 'pilatus']:
            self.executable = 'lmp_mpi'
            self.executable_opts = ['-in in.lj.cpu']
        else:
            self.executable = 'lmp_omp'
            self.executable_opts = ['-sf omp', '-pk omp 1', '-in in.lj.cpu']

        if self.scale == 'small':
            self.valid_systems += ['dom:mc']
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks_per_node = 36
            self.num_tasks = 576

        if self.current_system.name == 'eiger':
            self.num_tasks_per_node = 128
            self.num_tasks = 256 if self.scale == 'small' else 512

        self.reference = self.references_by_variant[self.variant][self.scale]
        self.tags |= {
            'maintenance' if self.variant == 'maint' else 'production'
        }
