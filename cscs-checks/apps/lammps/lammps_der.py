# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.base_check import LAMMPSBaseCheck


REFERENCE_ENERGY = {
    # every system has a different reference energy and drift
    'small': (-4.6195, 6.0E-04),
    'large': (-4.6195, 6.0E-04)
}

REFERENCE_GPU_PERFORMANCE = {
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

REFERENCE_CPU_PERFORMANCE = {
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
    },
}


class LAMMPSCheck(LAMMPSBaseCheck):
        strict_check = False
        extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

        tags = {'scs', 'external-resources'}
        maintainers = ['TR', 'VH']

        @run_after('init')
        def source_install(self):
            # Reset sources dir relative to the SCS apps prefix
            self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'LAMMPS')

        @run_after('init')
        def env_define(self):
                if self.current_system.name in ['eiger', 'pilatus']:
                    self.valid_prog_environs = ['cpeGNU']
                else:
                    self.valid_prog_environs = ['builtin']

        @run_after('init')
        def set_ref_tags(self):
            self.reference = self.references[self.variant][self.benchmark]
            self.tags |= {'maintenance' if self.variant == 'maint' else 'production'}


@rfm.simple_test
class LAMMPSGPUCheck(LAMMPSCheck):
        benchmark = parameter(['small', 'large'])
        variant = parameter(['prod', 'maint'])
        valid_systems = ['daint:gpu']
        executable = 'lmp_mpi'
        input_file = 'in.lj.gpu'
        executable_opts = ['-sf gpu', '-pk gpu 1', '-in', input_file]
        variables = {'CRAY_CUDA_MPS': '1'}
        num_gpus_per_node = 1
        references = REFERENCE_GPU_PERFORMANCE
        ener_ref = REFERENCE_ENERGY

        @run_after('init')
        def set_num_tasks(self):
                if self.benchmark == 'small':
                    self.valid_systems += ['dom:gpu']
                    self.num_tasks = 12
                    self.num_tasks_per_node = 2
                else:
                    self.num_tasks = 32
                    self.num_tasks_per_node = 2

@rfm.simple_test
class LAMMPSCPUCheck(LAMMPSCheck):
        variant = parameter(['prod'])
        valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
        benchmark = parameter(['small', 'large'])
        input_file = 'in.lj.cpu'
        references = REFERENCE_CPU_PERFORMANCE
        ener_ref = REFERENCE_ENERGY

        @run_after('init')
        def set_num_tasks(self):
            if self.benchmark == 'small':
                self.valid_systems += ['dom:mc']
                self.num_tasks = 216
                self.num_tasks_per_node = 36
            else:
                self.num_tasks_per_node = 36
                self.num_tasks = 576

            if self.current_system.name == 'eiger':
                self.num_tasks_per_node = 128
                self.num_tasks = 256 if self.benchmark == 'small' else 512


        @run_after('init')
        def set_hierarchical_prgenvs(self):
            if self.current_system.name in ['eiger', 'pilatus']:
                self.executable = 'lmp_mpi'
                self.executable_opts = ['-in', self.input_file]
            else:
                self.executable = 'lmp_omp'
                self.executable_opts = ['-sf omp', '-pk omp 1', '-in', self.input_file]
