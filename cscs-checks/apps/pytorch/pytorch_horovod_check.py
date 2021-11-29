# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import reframe as rfm
import reframe.utility.osext as osext

from hpctestlib.ml.pytorch.horovod import pytorch_cnn_check


@rfm.simple_test
class cscs_tensorflow_horovod_check(pytorch_cnn_check):
    num_nodes = parameter([8, 32, 1])
    model_name = parameter(['inception_v3', 'resnet50'])
    num_tasks_per_node = 1
    batch_size = 64
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']
    modules = ['PyTorch']
    tags |= {'production'}
    maintainers = ['sarafael', 'henrique']

    @run_after('init')
    def setup_filtering_criteria(self):
        self.model = self.model_name
        if self.num_nodes == 8:
            self.valid_systems += ['dom:gpu']

    @run_before('run')
    def setup_run(self):
        proc = self.current_partition.processor
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_cpus_per_task = proc.num_cores
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }

    # @run_before('performance')
    # def set_performance(self):
        ref_per_gpu_sm60 = 131 if self.model == 'inception_v3' else 201
        ref_total_sm60 = ref_per_gpu_sm60 * self.num_nodes
        allref = {
            'sm_60': {
                'throughput_total': (ref_total_sm60, -0.05, None, 'images/s'),
                'throughput_iteration': (ref_per_gpu_sm60, -0.05, None,
                                         'images/s')
            }
        }
        with contextlib.suppress(KeyError):
            self.reference = {
                '*': allref[self.num_nodes]['sm_60']
            }
