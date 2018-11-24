import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class GpuBurnTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.descr = 'GPU burn test'
        self.valid_prog_environs = ['PrgEnv-gnu']

        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['craype-accel-nvidia35']
            # NOTE: The first option indicates the precision (-d for double)
            #       while the seconds is the time (in secs) to run the test.
            #       For multi-gpu nodes, we run the gpu burn test for more
            #       time to get reliable measurements.
            self.executable_opts = ['-d', '40']
            self.num_gpus_per_node = 16
            gpu_arch = '37'
        else:
            self.modules = ['craype-accel-nvidia60']
            self.executable_opts = ['-d', '20']
            self.num_gpus_per_node = 1
            gpu_arch = '60'

        self.sourcepath = 'gpu_burn.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-arch=compute_%s' % gpu_arch,
                                      '-code=sm_%s' % gpu_arch]
        self.build_system.ldflags = ['-lcuda', '-lcublas', '-lnvidia-ml']

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall('OK', self.stdout)), self.num_tasks_assigned)

        self.perf_patterns = {
            'perf': sn.min(sn.extractall(
                r'GPU\s+\d+\(\S*\): (?P<perf>\S*) GF\/s', self.stdout,
                'perf', float))
        }

        self.reference = {
            'dom:gpu': {
                'perf': (4115, -0.10, None)
            },
            'daint:gpu': {
                'perf': (4115, -0.10, None)
            },
            'kesch:cn': {
                'perf': (950, -0.10, None)
            }
        }

        self.num_tasks = 0
        self.num_tasks_per_node = 1

        self.maintainers = ['AJ', 'VK', 'TM']

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks * self.num_gpus_per_node
