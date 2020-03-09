# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16')
@rfm.parameterized_test(['small'], ['large'])
class TensorFlowHorovodTest(rfm.RunOnlyRegressionTest):
    def __init__(self, variant):
        self.descr = 'Distributed training with TensorFlow and Horovod'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['builtin']
        tfshortver = '1.14'
        self.sourcesdir = 'https://github.com/tensorflow/benchmarks'
        self.modules = ['Horovod/0.16.4-CrayGNU-19.10-tf-%s.0' % tfshortver]
        if variant == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 8
            self.reference = {
                'dom:gpu': {
                    'throughput': (1133.6, -0.05, None, 'images/s'),
                },
                'daint:gpu': {
                    'throughput': (1134.8, -0.05, None, 'images/s')
                },
            }
        else:
            self.num_tasks = 32
            self.reference = {
                'daint:gpu': {
                    'throughput': (4403.0, -0.05, None, 'images/s')
                },
            }

        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        self.perf_patterns = {
            'throughput': sn.avg(sn.extractall(
                r'total images/sec:\s+(?P<throughput>\S+)',
                self.stdout, 'throughput', float))
        }

        self.sanity_patterns = sn.assert_eq(sn.count(sn.findall(
            r'total images/sec:', self.stdout)), self.num_tasks)

        self.pre_run = ['git checkout cnn_tf_v%s_compatible' % tfshortver]
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        self.executable = 'python'
        self.executable_opts = [
            'scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py',
            '--model inception3',
            '--batch_size 64',
            '--variable_update horovod',
            '--log_dir ./logs',
            '--train_dir ./checkpoints']
        self.tags = {'production'}
        self.maintainers = ['RS', 'TR']
