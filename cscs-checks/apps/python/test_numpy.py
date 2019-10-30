import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16')
@rfm.simple_test
class NumpyTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Test a few common numpy operations'
        self.valid_systems = ['dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['numpy/1.17.2-CrayGNU-19.10']
        self.reference = {
            'dom:gpu': {
                'dot': (0.66, None, 0.05, 'seconds'),
                'svd': (0.48, None, 0.05, 'seconds'),
                'cholesky': (0.14, None, 0.05, 'seconds'),
                'eigendec': (4.66, None, 0.05, 'seconds'),
                'inv': (0.26, None, 0.05, 'seconds'),
            },
        }
        self.perf_patterns = {
            'dot': sn.extractsingle(
                r'^Dotted two 4096x4096 matrices in\s+(?P<dot>\S+)\s+s',
                self.stdout, 'dot', float),
            'svd': sn.extractsingle(
                r'^SVD of a 2048x1024 matrix in\s+(?P<svd>\S+)\s+s',
                self.stdout, 'svd', float),
            'cholesky': sn.extractsingle(
                r'^Cholesky decomposition of a 2048x2048 matrix in'
                r'\s+(?P<cholesky>\S+)\s+s',
                self.stdout, 'cholesky', float),
            'eigendec': sn.extractsingle(
                r'^Cholesky decomposition of a 2048x2048 matrix in'
                r'\s+(?P<eigendec>\S+)\s+s',
                self.stdout, 'eigendec', float),
            'inv': sn.extractsingle(
                r'^Inversion of a 2048x2048 matrix in\s+(?P<inv>\S+)\s+s',
                self.stdout, 'inv', float)
        }

        np_version = sn.extractsingle(r'Numpy version:\s+(?P<ver>\S+)',
                                      self.stdout, 'ver', str)
        self.sanity_patterns = sn.assert_eq(np_version, '1.17.2')
        self.variables = {
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        self.executable = 'python'
        self.executable_opts = ['np_test.py']
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        self.tags = {'production'}
        self.maintainers = ['RS', 'TR']
