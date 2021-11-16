# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import reframe as rfm
import reframe.utility.osext as osext

from hpctestlib.ml.tensorflow.horovod import tensorflow_cnn_check


@rfm.simple_test
class cscs_tensorflow_horovod_check(tensorflow_cnn_check):
    num_nodes = parameter([8, 32])
    num_tasks_per_node = 1
    batch_size = 64
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']
    modules = [
        f'Horovod/0.21.0-CrayGNU-{osext.cray_cdt_version()}-tf-2.4.0'
    ]
    tags |= {'production'}
    maintainers = ['sarafael', 'henrique']
    allref = {
        8: {
            'sm_60': {
                'throughput_total': (1712, -0.05, None, 'images/s'),
                'throughput_iteration': (214, -0.05, None, 'images/s')
            }
        },
        16: {
            'sm_60': {
                'throughput_total': (6848, -0.05, None, 'images/s'),
                'throughput_iteration': (214, -0.05, None, 'images/s')
            }
        },
    }

    @run_after('init')
    def setup_filtering_criteria(self):
        if self.num_nodes == 32:
            self.valid_systems += ['dom:gpu']

    @run_before('run')
    def setup_run(self):
        proc = self.current_partition.processor
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_cpus_per_task = proc.num_cores
        with contextlib.suppress(KeyError):
            self.reference = {
                '*': self.allref[self.num_nodes]['sm_60']
            }

        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
