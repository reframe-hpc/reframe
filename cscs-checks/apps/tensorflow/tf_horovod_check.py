import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class TensorFlowHorovodTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Distributed training with TensorFlow and Horovod'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        tfshortver = '1.11'
        self.sourcesdir = 'https://github.com/tensorflow/benchmarks'
        self.modules = ['Horovod/0.15.0-CrayGNU-18.08-tf-%s.0' % tfshortver]
        self.reference = {
            'dom:gpu': {
                'img_sec': (1133.6, None, 0.05),
            },
            'daint:gpu': {
                'img_sec': (4403.0, None, 0.05)
            },
        }
        self.perf_patterns = {
            'img_sec': sn.avg(sn.extractall(
                r'total images/sec:\s+(?P<img_sec>\S+)',
                self.stdout, 'img_sec', float))
        }
        self.sanity_patterns = sn.assert_found(
            r'[\S+\s+] INFO NET\/IB : Using interface ipogif0'
            r' for sideband communication', self.stdout)
        self.num_tasks_per_node = 1
        if self.current_system.name == 'dom':
            self.num_tasks = 8

        if self.current_system.name == 'daint':
            self.num_tasks = 32

        self.pre_run = ['git checkout cnn_tf_v%s_compatible' % tfshortver]
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        self.executable = ('python scripts/tf_cnn_benchmarks/'
                           'tf_cnn_benchmarks.py')
        self.executable_opts = [
            '--model inception3',
            '--batch_size 64',
            '--variable_update horovod',
            '--log_dir ./logs',
            '--train_dir ./checkpoints']
        self.tags = {'production'}
        self.maintainers = ['MS', 'RS']
