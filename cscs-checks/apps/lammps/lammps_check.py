# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.lammps.nve import LAMMPS_NVE

dom_gpu_small = {
    'maint': (3457, -0.10, None, 'timesteps/s'),
    'prod': (3132, -0.05, None, 'timesteps/s'),
}

daint_gpu_small = {
    'maint': (2524, -0.10, None, 'timesteps/s'),
    'prod': (2400, -0.40, None, 'timesteps/s'),
}

REFERENCE_GPU_PERFORMANCE_SMALL = {
    'dom:gpu': dom_gpu_small,
    'daint:gpu': daint_gpu_small,
}


daint_gpu_large = {
    'maint': (3832, -0.05, None, 'timesteps/s'),
    'prod': (3260, -0.50, None, 'timesteps/s'),
}

REFERENCE_GPU_PERFORMANCE_LARGE = {
    'daint:gpu': daint_gpu_large,
}

dom_cpu_small = {
    'prod': (4394, -0.05, None, 'timesteps/s'),
}

daint_cpu_small = {
    'prod': (3824, -0.10, None, 'timesteps/s'),
}

eiger_cpu_small = {
    'prod': (4500, -0.10, None, 'timesteps/s'),
}

pilatus_cpu_small = {
    'prod': (5000, -0.10, None, 'timesteps/s'),
}

REFERENCE_CPU_PERFORMANCE_SMALL = {
    'dom:mc': dom_cpu_small,
    'daint:mc': daint_cpu_small,
    'eiger:mc': eiger_cpu_small,
    'pilatus:mc': pilatus_cpu_small

}

daint_cpu_large = {
    'prod': (5310, -0.65, None, 'timesteps/s'),
}

eiger_cpu_large = {
    'prod': (6500, -0.10, None, 'timesteps/s'),
}

pilatus_cpu_large = {
    'prod': (7500, -0.10, None, 'timesteps/s'),
}

REFERENCE_CPU_PERFORMANCE_LARGE = {
    'daint:mc': daint_cpu_large,
    'eiger:mc': eiger_cpu_large,
    'pilatus:mc': pilatus_cpu_large,

}


def inherit_cpu_only(params):
    return tuple(filter(lambda p: p[0] == 'cpu', params))


def inherit_gpu_only(params):
    return tuple(filter(lambda p: p[0] == 'gpu', params))


class LAMMPSCheckCSCS(LAMMPS_NVE):
    modules = ['LAMMPS']
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    tags = {'scs', 'external-resources'}
    maintainers = ['TR', 'VH']

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('init')
    def set_tags(self):
        self.tags |= {'maintenance' if self.mode == 'maint'
                      else 'production'}


@rfm.simple_test
class lammps_gpu_check(LAMMPSCheckCSCS):
    platform = parameter(inherit_params=True,
                         filter_params=inherit_gpu_only)
    scale = parameter(['small', 'large'])
    mode = parameter(['prod', 'maint'])
    valid_systems = ['daint:gpu']
    num_gpus_per_node = 1

    @run_after('init')
    def set_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_GPU_PERFORMANCE_SMALL
        else:
            self.reference = REFERENCE_GPU_PERFORMANCE_LARGE

    @run_before('run')
    def set_executable_opts(self):
        self.executable = 'lmp_mpi'
        self.executable_opts = ['-sf gpu',
                                '-pk gpu 1',
                                '-in', self.input_file]
        self.variables = {'CRAY_CUDA_MPS': '1'}
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 12
            self.num_tasks_per_node = 2
        else:
            self.num_tasks = 32
            self.num_tasks_per_node = 2


@rfm.simple_test
class lammps_cpu_check(LAMMPSCheckCSCS):
    platform = parameter(inherit_params=True,
                         filter_params=inherit_cpu_only)
    scale = parameter(['small', 'large'])
    mode = parameter(['prod'])
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']

    @run_after('init')
    def set_num_tasks(self):
        if self.scale == 'small':
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
    def set_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_CPU_PERFORMANCE_SMALL
        else:
            self.reference = REFERENCE_CPU_PERFORMANCE_LARGE

    @run_before('run')
    def set_hierarchical_prgenvs(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.executable = 'lmp_mpi'
            self.executable_opts = ['-in', self.input_file]
        else:
            self.executable = 'lmp_omp'
            self.executable_opts = ['-sf omp',
                                    '-pk omp 1',
                                    '-in', self.input_file]
