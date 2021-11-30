# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import reframe as rfm

from hpctestlib.ml.pytorch.horovod import pytorch_cnn_check


@rfm.simple_test
class cscs_pytorch_horovod_check(pytorch_cnn_check):
    num_nodes = parameter([1, 8, 32])
    num_tasks_per_node = 1
    batch_size = 64
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']
    modules = ['PyTorch']
    tags |= {'production'}
    maintainers = ['sarafael', 'henrique']
    allref = {
        'sm_60': {
            'inception_v3': {
                'throughput_per_gpu': (131, -0.05, None, 'images/s'),
            },
            'resnet50': {
                'throughput_per_gpu': (201, -0.05, None, 'images/s'),
            }
        }
    }
    model_name = parameter(allref['sm_60'].keys())

    @run_after('init')
    def setup_filtering_criteria(self):
        self.model = self.model_name
        if self.num_nodes == 8:
            self.valid_systems += ['dom:gpu']

    @run_before('run')
    def setup_run(self):
        self.skip_if_no_procinfo()
        proc = self.current_partition.processor
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_cpus_per_task = proc.num_cores
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        ref_per_gpu = self.allref['sm_60'][self.model]['throughput_per_gpu'][0]
        ref_total = ref_per_gpu * self.num_nodes
        with contextlib.suppress(KeyError):
            self.reference = {
                '*': {
                    **self.allref['sm_60'][self.model],
                    'throughput_total': (ref_total, -0.05, None, 'images/s'),
                }
            }
