# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.vasp.base_check import VASP

dom_cpu = {
    'maint': (148.7, None, 0.05, 's'),
    'prod': (148.7, None, 0.05, 's'),
}

daint_cpu = {
    'maint': (105.3, None, 0.20, 's'),
    'prod': (105.3, None, 0.20, 's'),
}

eiger_cpu = {
    'maint': (100.0, None, 0.10, 's'),
    'prod': (100.0, None, 0.10, 's'),
}

pilatus_cpu = {
    'maint': (100.0, None, 0.10, 's'),
    'prod': (100.0, None, 0.10, 's'),
}


REFERENCE_CPU_PERFORMANCE = {
    'dom:gpu': dom_cpu,
    'daint:gpu': daint_cpu,
    'eiger:mc': eiger_cpu,
    'pilatus:mc': pilatus_cpu
}


dom_gpu = {
    'maint': (61.0, None, 0.10, 's'),
    'prod': (46.7, None, 0.20, 's'),
}

daint_gpu = {
    'maint': (46.7, None, 0.20, 's'),
    'prod': (46.7, None, 0.20, 's'),
}

REFERENCE_GPU_PERFORMANCE = {
    'dom:gpu': dom_gpu,
    'daint:gpu': daint_gpu,
}

REFERENCE_PERFORMANCE = {
    'gpu': REFERENCE_GPU_PERFORMANCE,
    'cpu': REFERENCE_CPU_PERFORMANCE,
}


@rfm.simple_test
class VASPCheckCSCS(VASP):
    mode = parameter(['prod', 'maint'])
    modules = ['VASP']
    maintainers = ['LM']
    tags = {'scs'}
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('init')
    def set_tags(self):
        self.tags |= {'maintenance' if self.mode == 'maint'
                      else 'production'}

    @run_after('init')
    def set_valid_systems(self):
        if self.platform_info[0] == 'cpu':
            self.valid_systems = ['daint:mc',
                                  'dom:mc',
                                  'eiger:mc',
                                  'pilatus:mc']
        else:
            self.valid_systems = ['daint:gpu', 'dom:gpu']

    @run_after('setup')
    def set_num_tasks(self):
        if self.platform == 'cpu':
            if self.current_system.name == 'dom':
                self.num_tasks = 72
                self.num_tasks_per_node = 12
                self.use_multithreading = True
            elif self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 64
                self.num_tasks_per_node = 4
                self.num_cpus_per_task = 8
                self.num_tasks_per_core = 1
                self.use_multithreading = False
                self.variables = {
                    'MPICH_OFI_STARTUP_CONNECT': '1',
                    'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }
            else:
                self.num_tasks = 32
                self.num_tasks_per_node = 2
                self.use_multithreading = True
        else:
            self.variables = {'CRAY_CUDA_MPS': '1'}
            self.num_gpus_per_node = 1
            if self.current_system.name == 'dom':
                self.num_tasks = 6
                self.num_tasks_per_node = 1
            else:
                self.num_tasks = 16
                self.num_tasks_per_node = 1

    @run_after('setup')
    def set_perf_reference(self):
        self.reference = REFERENCE_PERFORMANCE[self.platform]

    @run_after('setup')
    def set_description(self):
        self.descr = f'VASP {self.platform} check (benchmark: {self.mode})'

    @run_before('run')
    def set_task_distribution(self):
        if self.platform == 'cpu':
            self.job.options = ['--distribution=block:block']

    @run_before('run')
    def set_cpu_binding(self):
        if self.platform == 'cpu':
            self.job.launcher.options = ['--cpu-bind=cores']
