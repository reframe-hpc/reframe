# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*[[model, mpi_task]
                          for model in ['inception_v3']
                          for mpi_task in [2]
                          # TODO: for model in ['inception_v3', 'resnet50']
                          # TODO: for mpi_task in [1, 2, ...]
                          ])
class PytorchHorovodTest(rfm.RunOnlyRegressionTest):
    def __init__(self, model, mpi_task):
        self.descr = f'Distributed training with Pytorch and Horovod'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['builtin']
        self.modules = ['Horovod/0.19.5-CrayGNU-20.08-pt-1.6.0']
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
        git_url = (f'https://raw.githubusercontent.com/horovod/horovod/{hash}/'
                   'examples/pytorch')
        git_src = 'pytorch_synthetic_benchmark.py'
        # this if will be removed after horovod will be built with scipy
        if model in ['inception_v3']:
            self.prerun_cmds = [
                'python3 -m venv --system-site-packages myvenv',
                'source myvenv/bin/activate',
                'pip install scipy']

        self.prerun_cmds += [
            f'wget {git_url}/{git_src}',
            'sed -i "s-output = model(data)-output, aux = model(data)-"'
            f' {git_src}',
            'sed -i "s-data = torch.randn(args.batch_size, 3, 224, 224)-'
            f'data = torch.randn(args.batch_size, 3, 299, 299)-" {git_src}',
            'echo starttime=`date +%s`']
        self.postrun_cmds = ['echo stoptime=`date +%s`']
        self.executable = 'python'
        self.executable_opts = [
            git_src, f'--model {model}',
            f'--batch-size {batch_size}', '--num-iters 5',
            '--num-batches-per-iter 5', '--num-warmup-batches 5']
        self.tags = {'production'}
        self.maintainers = ['RS', 'HM']
        self.sanity_patterns = sn.all([
            sn.assert_found(rf'Model: {model}', self.stdout),
            sn.assert_found(rf'Batch size: {batch_size}', self.stdout)
        ])
        regex_start_sec = r'^starttime=(?P<sec>\d+.\d+)'
        regex_stop_sec = r'^stoptime=(?P<sec>\d+.\d+)'
        start_sec = sn.extractsingle(regex_start_sec, self.stdout, 'sec', int)
        stop_sec = sn.extractsingle(regex_stop_sec, self.stdout, 'sec', int)
        self.perf_patterns = {
            'elapsed': stop_sec - start_sec,
            'throughput_per_gpu': sn.extractsingle(
                r'Img/sec per GPU: (?P<throughput_per_gpu>\S+) \S+',
                self.stdout, 'throughput_per_gpu', float),
            'throughput_per_job': sn.extractsingle(
                r'Total img/sec on \d+ GPU\(s\): (?P<throughput>\S+) \S+',
                self.stdout, 'throughput', float),
        }
        ref_per_gpu = 131.
        ref_per_job = ref_per_gpu * mpi_task
        self.reference = {
            'dom:gpu': {
                'elapsed': (0, None, None, 's'),
                'throughput_per_gpu': (ref_per_gpu, -0.05, None, 'images/s'),
                'throughput_per_job': (ref_per_job, -0.05, None, 'images/s'),
            },
            'daint:gpu': {
                'elapsed': (0, None, None, 's'),
                'throughput_per_gpu': (ref_per_gpu, -0.05, None, 'images/s'),
                'throughput_per_job': (ref_per_job, -0.05, None, 'images/s'),
            },
        }
