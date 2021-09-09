# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.pytorch.base_check import PytorchHorovod_BaseTest


@rfm.simple_test
class PytorchHorovodTest(PytorchHorovod_BaseTest):
    tags = {'production'}
    maintainers = ['RS', 'HM']
    mpi_task = parameter([32, 8, 1])
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']
    modules = ['PyTorch']
    num_tasks_per_node = 1
    num_cpus_per_task = 12
    batch_size = 64

    @run_after('init')
    def set_valid_systems(self):
        self.num_tasks = self.mpi_task
        if self.mpi_task < 20:
            self.valid_systems += ['dom:gpu']

    @run_after('init')
    def set_variables(self):
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }

    @run_after('setup')
    def set_reference(self):
        ref_per_gpu = 131 if self.model == 'inception_v3' else 201
        ref_per_job = ref_per_gpu * self.mpi_task
        self.reference = {
            'dom:gpu': {
                'throughput_per_gpu': (ref_per_gpu, -0.1, None, 'images/s'),
                'throughput_per_job': (ref_per_job, -0.1, None, 'images/s'),
            },
            'daint:gpu': {
                'throughput_per_gpu': (ref_per_gpu, -0.1, None, 'images/s'),
                'throughput_per_job': (ref_per_job, -0.1, None, 'images/s'),
            }
        }
