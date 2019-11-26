import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16')
@rfm.parameterized_test(['small'], ['large'])
class TensorFlow2HorovodTest(rfm.RunOnlyRegressionTest):
    def __init__(self, variant):
        self.descr = 'Distributed training with TensorFlow2 and Horovod'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['Horovod/0.18.1-CrayGNU-19.10-tf-2.0.0']
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        if variant == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 8
            self.reference = {
                'dom:gpu': {
                    'throughput': (2031.6, -0.05, None, 'images/s'),
                    'throughput_per_gpu': (253.9, -0.05, None, 'images/s'),
                },
                'daint:gpu': {
                    'throughput': (2031.6, -0.05, None, 'images/s'),
                    'throughput_per_gpu': (253.9, -0.05, None, 'images/s')
                },
            }
        else:
            self.num_tasks = 32
            self.reference = {
                'daint:gpu': {
                    'throughput': (7976.6, -0.05, None, 'images/s'),
                    'throughput_per_gpu': (253.9, -0.05, None, 'images/s')
                },
            }
        self.perf_patterns = {
            'throughput': sn.extractsingle(
                r'Total img/sec on %s GPU\(s\): '
                r'(?P<throughput>\S+) \S+' % self.num_tasks,
                self.stdout, 'throughput', float),
            'throughput_per_gpu': sn.extractsingle(
                r'Img/sec per GPU: (?P<throughput_per_gpu>\S+) \S+',
                self.stdout, 'throughput_per_gpu', float)
        }
        model = 'InceptionV3'
        batch_size = 64
        self.sanity_patterns = sn.all([
            sn.assert_found(r'Model: %s' % model, self.stdout),
            sn.assert_found(r'Batch size: %s' % batch_size, self.stdout)
        ])
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_IB_HCA': 'ipogif0',
            'NCCL_IB_CUDA_SUPPORT': '1',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        self.pre_run = ['wget https://raw.githubusercontent.com/horovod/'
                        'horovod/26b55a7890f6923ca58cdb68a765ed0ec436ab0f/'
                        'examples/tensorflow2_synthetic_benchmark.py']
        self.executable = 'python'
        self.executable_opts = [
            'tensorflow2_synthetic_benchmark.py',
            '--model %s' % model,
            '--batch-size %s' % batch_size,
        ]
        self.tags = {'production'}
        self.maintainers = ['VK', 'RS']
