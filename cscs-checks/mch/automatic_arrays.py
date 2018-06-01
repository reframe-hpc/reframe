import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class AutomaticArraysCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('automatic_arrays_check',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray*', 'PrgEnv-pgi*',
                                    'PrgEnv-gnu']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self._pgi_flags = '-acc -ta=tesla:cc60'
        elif self.current_system.name in ['kesch']:
            self.modules = ['craype-accel-nvidia35']
            self._pgi_flags = '-O2 -ta=tesla,cc35,cuda8.0'

        self.num_tasks = 1
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcepath = 'automatic_arrays.f90'
        self.sanity_patterns = sn.assert_found(r'Result:\s+OK', self.stdout)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'Timing:\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float)
        }

        self.aarrays_reference = {
            'PrgEnv-cray': {
                'daint:gpu': {'perf': (1.4E-04, None, 0.15)},
                'dom:gpu': {'perf': (1.4E-04, None, 0.15)},
                'kesch:cn': {'perf': (2.9E-04, None, 0.15)},
            },
            'PrgEnv-gnu': {
                'daint:gpu': {'perf': (7.4E-03, None, 0.15)},
                'dom:gpu': {'perf': (7.4E-03, None, 0.15)},
                'kesch:cn': {'perf': (6.5E-03, None, 0.15)},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'perf': (1.4E-04, None, 0.15)},
                'dom:gpu': {'perf': (1.4E-04, None, 0.15)},
                'kesch:cn': {'perf': (1.4E-04, None, 0.15)},
            }
        }

        self.maintainers = ['AJ', 'VK']

    def setup(self, partition, environ, **job_opts):
        if 'PrgEnv-cray' in environ.name:
            environ.fflags = '-O2 -hacc -hnoomp'
        elif 'PrgEnv-pgi' in environ.name:
            environ.fflags = self._pgi_flags
        elif 'PrgEnv-gnu' in environ.name:
            environ.fflags = '-O2'

        super().setup(partition, environ, **job_opts)
        self.reference = self.aarrays_reference[self.current_environ.name]


def _get_checks(**kwargs):
    return [AutomaticArraysCheck(**kwargs)]
