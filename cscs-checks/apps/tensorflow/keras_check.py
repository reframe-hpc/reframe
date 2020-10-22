# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


imagenet_tar = '/apps/common/UES/reframe/resources/training/imagenet.tar'


@rfm.parameterized_test(*[[mpi_task, batch_size]
                          for mpi_task in [1]
                          for batch_size in [128]
                          ])
class KerasTest(rfm.RunOnlyRegressionTest):
    def __init__(self, mpi_task, batch_size):
        self.descr = f'Distributed training with TF and Keras'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['builtin']
        self.modules = ['TensorFlow/2.2.0-CrayGNU-20.08-cuda-10.1.168']
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        self.num_tasks = mpi_task
        self.variables = {
            # 'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        imagenet_tar_filename = imagenet_tar.split('/')[-1]
        self.prerun_cmds = [
            f'cp {imagenet_tar} .', f"tar xf {imagenet_tar_filename}",
            f'rm {imagenet_tar_filename}',
            'nvidia-smi -q -d ACCOUNTING,MEMORY',
            'echo starttime=`date +%s`']
        self.postrun_cmds = ['echo stoptime=`date +%s`']
        self.rpt = 'rpt'
        self.executable = './01_inceptionv3_tfr.py'
        self.executable_opts = [str(batch_size), '"imagenet/train-*"']
        self.tags = {'production'}
        self.maintainers = ['RS', 'HM', 'JG']
        self.sanity_patterns = sn.all([
            sn.assert_found(r'batch_size=', self.stdout),
            sn.assert_found(r'accuracy: ', self.stdout),
        ])
        # ms/step: "51s 514ms/step - loss: 7.0166 - accuracy: 0.0012"
        regex = (r'100\/(?P<steps>\d+).* - (?P<sec>\d+)s (?P<ms_stp>\d+)ms\/'
                 r'step - loss: (?P<loss>\S+) - accuracy: (?P<accuracy>\S+)')
        ms_per_step = sn.extractsingle(regex, self.stdout, 'ms_stp', int)
        accuracy = sn.extractsingle(regex, self.stdout, 'accuracy', float)
        # walltime:
        regex_start_sec = r'^starttime=(?P<sec>\d+.\d+)'
        regex_stop_sec = r'^stoptime=(?P<sec>\d+.\d+)'
        start_sec = sn.extractsingle(regex_start_sec, self.stdout, 'sec', int)
        stop_sec = sn.extractsingle(regex_stop_sec, self.stdout, 'sec', int)
        # Total GPU Memory reported by nvidia-smi:
        regex_hw = r'^\s+Total\s+:\s+(?P<memT>\d+) MiB'
        mem_per_gpu = sn.extractsingle(regex_hw, self.stdout, 'memT', int)
        # Min/Avg/Max GPU Utilization reported by nvidia-smi:
        regex_acctg = (r'Tesla.*, (?P<pid>\d+), (?P<gpu_use>\d+) %, \d+ %, '
                       r'(?P<max_mem>\d+) MiB, (?P<ms>\d+) ')
        min_gpu_use = sn.min(sn.extractall(regex_acctg, self.rpt, 'gpu_use',
                                           int))
        avg_gpu_use = sn.round(sn.avg(sn.extractall(regex_acctg, self.rpt,
                                                    'gpu_use', int)), 0)
        max_gpu_use = sn.max(sn.extractall(regex_acctg, self.rpt, 'gpu_use',
                                           int))
        # Min/Avg/Max GPU Memory Utilization reported by nvidia-smi:
        min_gpu_mem = sn.min(sn.extractall(regex_acctg, self.rpt, 'max_mem',
                                           int))
        avg_gpu_mem = sn.round(sn.avg(sn.extractall(regex_acctg, self.rpt,
                                                    'max_mem', int)), 0)
        max_gpu_mem = sn.max(sn.extractall(regex_acctg, self.rpt, 'max_mem',
                                           int))
        self.perf_patterns = {
            'elapsed': stop_sec - start_sec,
            'throughput': ms_per_step,
            'accuracy': accuracy,
            'min_gpu_use': min_gpu_use,
            'avg_gpu_use': avg_gpu_use,
            'max_gpu_use': max_gpu_use,
            'min_gpu_mem': sn.round(min_gpu_mem / mem_per_gpu * 100, 0),
            'avg_gpu_mem': sn.round(avg_gpu_mem / mem_per_gpu * 100, 0),
            'max_gpu_mem': sn.round(max_gpu_mem / mem_per_gpu * 100, 0),
        }
        ref_per_gpu = 787.
        self.reference = {
            'dom:gpu': {
                'elapsed': (0, None, None, 's'),
                'throughput': (ref_per_gpu, None, 0.05, 'ms/step'),
                'accuracy': (0.002, -0.05, None, ''),
                'min_gpu_use': (0, None, None, '%'),
                'avg_gpu_use': (0, None, None, '%'),
                'max_gpu_use': (0, None, None, '%'),
                'min_gpu_mem': (0, None, None, '%'),
                'avg_gpu_mem': (0, None, None, '%'),
                'max_gpu_mem': (0, None, None, '%'),
            },
            'daint:gpu': {
                'elapsed': (0, None, None, 's'),
                'throughput': (ref_per_gpu, None, 0.05, 'ms/step'),
                'accuracy': (0.002, -0.05, None, ''),
                'min_gpu_use': (0, None, None, '%'),
                'avg_gpu_use': (0, None, None, '%'),
                'max_gpu_use': (0, None, None, '%'),
                'min_gpu_mem': (0, None, None, '%'),
                'avg_gpu_mem': (0, None, None, '%'),
                'max_gpu_mem': (0, None, None, '%'),
            },
        }

    @rfm.run_before('run')
    def set_launcher(self):
        accounting_script = './usage.sh'
        self.postrun_cmds += [
            f'{self.job.launcher.command(self.job)[0]} {accounting_script}',
            f'tail -q -n 1 *.csv > {self.rpt}',
            'rm -fr imagenet',
        ]
