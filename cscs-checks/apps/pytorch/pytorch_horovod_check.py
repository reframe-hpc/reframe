# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext


@rfm.parameterized_test(*[[model, mpi_task]
                          for mpi_task in [32, 8, 1]
                          for model in ['inception_v3', 'resnet50']])
class PytorchHorovodTest(rfm.RunOnlyRegressionTest):
    def __init__(self, model, mpi_task):
        self.descr = 'Distributed training with Pytorch and Horovod'
        self.valid_systems = ['daint:gpu']
        if mpi_task < 20:
            self.valid_systems += ['dom:gpu']

        self.valid_prog_environs = ['builtin']
        cray_cdt_version = osext.cray_cdt_version()
        self.modules = [f'Horovod/0.19.5-CrayGNU-{cray_cdt_version}-pt-1.6.0']
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        self.num_tasks = mpi_task
        batch_size = 64
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        hash = 'master'
        git_url = f'https://raw.githubusercontent.com/horovod/horovod/{hash}/examples/pytorch'  # noqa: E501
        git_src = 'pytorch_synthetic_benchmark.py'
        self.prerun_cmds = [f'wget {git_url}/{git_src}']

        if model == 'inception_v3':
            self.prerun_cmds += [
                'python3 -m venv --system-site-packages myvenv',
                'source myvenv/bin/activate',
                'pip install scipy',
                'sed -i "s-output = model(data)-output, aux = model(data)-"'
                f' {git_src}',
                'sed -i "s-data = torch.randn(args.batch_size, 3, 224, 224)-'
                f'data = torch.randn(args.batch_size, 3, 299, 299)-"'
                f' {git_src}'
            ]

        self.executable = 'python'
        self.executable_opts = [
            git_src,
            f'--model {model}',
            f'--batch-size {batch_size}',
            '--num-iters 5',
            '--num-batches-per-iter 5'
        ]
        self.tags = {'production'}
        self.maintainers = ['RS', 'HM']
        self.sanity_patterns = sn.all([
            sn.assert_found(rf'Model: {model}', self.stdout),
            sn.assert_found(rf'Batch size: {batch_size}', self.stdout)
        ])
        self.perf_patterns = {
            'throughput_per_gpu': sn.extractsingle(
                r'Img/sec per GPU: (?P<throughput_per_gpu>\S+) \S+',
                self.stdout, 'throughput_per_gpu', float
            ),
            'throughput_per_job': sn.extractsingle(
                r'Total img/sec on \d+ GPU\(s\): (?P<throughput>\S+) \S+',
                self.stdout, 'throughput', float
            ),
        }
        ref_per_gpu = 131 if model == 'inception_v3' else 201
        ref_per_job = ref_per_gpu * mpi_task
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
