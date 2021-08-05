# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.vasp import VASP

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


class VASPCheck(VASP):
    modules = ['VASP']
    maintainers = ['LM']
    tags = {'scs'}
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    force_value = -.85026214E+03
    force_tolerance = 1e-5

    @run_after('setup')
    def set_generic_perf_references(self):
        self.reference.update({'*': {
            self.benchmark: (0, None, None, 's')
        }})

    @run_after('setup')
    def set_perf_patterns(self):
        self.perf_patterns = {
            self.benchmark: sn.extractsingle(r'Total CPU time used \(sec\):'
                                             r'\s+(?P<time>\S+)', 'OUTCAR',
                                             'time', float)
        }

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('init')
    def set_tags(self):
        self.tags |= {'maintenance' if self.benchmark == 'maint'
                      else 'production'}


@rfm.simple_test
class VASPCpuCheck(VASPCheck):
    valid_systems = ['daint:mc', 'dom:mc', 'eiger:mc', 'pilatus:mc']
    executable = 'vasp_std'
    reference = REFERENCE_CPU_PERFORMANCE
    benchmark = parameter(['prod', 'maint'])

    @run_after('init')
    def set_num_tasks(self):
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

    @run_after('init')
    def set_description(self):
        self.descr = f'VASP CPU check (benchmark: {self.benchmark})'

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']


@rfm.simple_test
class VASPGpuCheck(VASPCheck):
    valid_systems = ['daint:gpu', 'dom:gpu']
    executable = 'vasp_gpu'
    reference = REFERENCE_GPU_PERFORMANCE
    benchmark = parameter(['prod', 'maint'])
    variables = {'CRAY_CUDA_MPS': '1'}
    num_gpus_per_node = 1

    @run_after('init')
    def set_num_tasks(self):
        if self.current_system.name == 'dom':
            self.num_tasks = 6
            self.num_tasks_per_node = 1
        else:
            self.num_tasks = 16
            self.num_tasks_per_node = 1

    @run_after('init')
    def set_description(self):
        self.descr = f'VASP GPU check (benchmark: {self.benchmark})'
