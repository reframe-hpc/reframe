# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
from hpctestlib.apps.tensorflow.base_check import TensorFlow2Horovod_BaseTest


REFERENCE_SMALL_PERFOMANCE = {
    'dom:gpu': {
        'throughput': (1712, -0.05, None, 'images/s'),
        'throughput_per_gpu': (214, -0.05, None, 'images/s'),
    },
    'daint:gpu': {
        'throughput': (1712, -0.05, None, 'images/s'),
        'throughput_per_gpu': (214, -0.05, None, 'images/s')
    },
}

REFERENCE_LARGE_PERFOMANCE = {
    'daint:gpu': {
        'throughput': (6848, -0.05, None, 'images/s'),
        'throughput_per_gpu': (214, -0.05, None, 'images/s')
    },
}


@rfm.simple_test
class TensorFlow2HorovodTestCSCS(TensorFlow2Horovod_BaseTest):
    variant = parameter(['small', 'large'])
    sourcesdir = None
    num_tasks_per_node = 1
    num_cpus_per_task = 12
    batch_size = 64
    tags = {'production'}
    maintainers = ['RS', 'TR']
    valid_prog_environs = ['builtin']
    valid_systems = ['daint:gpu']
    modules = [
        f'Horovod/0.21.0-CrayGNU-{osext.cray_cdt_version()}-tf-2.4.0'
    ]

    @run_after('init')
    def set_num_task(self):
        if self.variant == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 8
            self.reference = REFERENCE_SMALL_PERFOMANCE
        else:
            self.num_tasks = 32
            self.reference = REFERENCE_LARGE_PERFOMANCE

    @run_after('init')
    def set_executable_opts(self):
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        self.executable_opts = [
            f'{self.script}',
            f'--model {self.model}',
            f'--batch-size {self.batch_size}',
            '--num-iters 5',
            '--num-batches-per-iter 5',
            '--num-warmup-batches 5',
        ]
