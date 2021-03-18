# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext


@rfm.simple_test
class TensorFlow2HorovodTest(rfm.RunOnlyRegressionTest):
    variant = parameter(['small', 'large'])

    def __init__(self):
        self.descr = 'Distributed training with TensorFlow2 and Horovod'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['builtin']
        self.modules = [
            f'Horovod/0.21.0-CrayGNU-{osext.cray_cdt_version()}-tf-2.4.0'
        ]
        self.sourcesdir = None
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        if self.variant == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 8
            self.reference = {
                'dom:gpu': {
                    'throughput': (1712, -0.05, None, 'images/s'),
                    'throughput_per_gpu': (214, -0.05, None, 'images/s'),
                },
                'daint:gpu': {
                    'throughput': (1712, -0.05, None, 'images/s'),
                    'throughput_per_gpu': (214, -0.05, None, 'images/s')
                },
            }
        else:
            self.num_tasks = 32
            self.reference = {
                'daint:gpu': {
                    'throughput': (6848, -0.05, None, 'images/s'),
                    'throughput_per_gpu': (214, -0.05, None, 'images/s')
                },
            }
        self.perf_patterns = {
            'throughput': sn.extractsingle(
                rf'Total img/sec on {self.num_tasks} GPU\(s\): '
                rf'(?P<throughput>\S+) \S+',
                self.stdout, 'throughput', float),
            'throughput_per_gpu': sn.extractsingle(
                r'Img/sec per GPU: (?P<throughput_per_gpu>\S+) \S+',
                self.stdout, 'throughput_per_gpu', float)
        }
        model = 'InceptionV3'
        batch_size = 64
        self.sanity_patterns = sn.all([
            sn.assert_found(rf'Model: {model}', self.stdout),
            sn.assert_found(rf'Batch size: {batch_size}', self.stdout)
        ])
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        script = 'tensorflow2_synthetic_benchmark.py'
        self.prerun_cmds = ['wget https://raw.githubusercontent.com/horovod/'
                            'horovod/842d1075e8440f15e84364f494645c28bf20c3ae/'
                            'examples/tensorflow2_synthetic_benchmark.py',
                            'sed -i "s/weights=None/weights=None, '
                            f'input_shape=(224, 224, 3)/g" {script}']
        self.executable = 'python'
        self.executable_opts = [
            f'{script}',
            f'--model {model}',
            f'--batch-size {batch_size}',
            '--num-iters 5',
            '--num-batches-per-iter 5',
            '--num-warmup-batches 5',
        ]
        self.tags = {'production'}
        self.maintainers = ['RS', 'TR']
