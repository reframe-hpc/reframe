# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class IPCMagicCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Distributed training with TensorFlow using ipyparallel'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = [
            # FIXME: Use the default ipcmagic version when fixed
            f'ipcmagic/0.1-CrayGNU-{osext.cray_cdt_version()}',
            f'Horovod/0.21.0-CrayGNU-{osext.cray_cdt_version()}-tf-2.4.0'
        ]
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.executable = 'ipython'
        self.executable_opts = ['tf-hvd-sgd-ipc-tf2.py']
        nids = sn.extractall(r'nid(?P<nid>\d+)',
                             self.stdout, 'nid', str)
        self.sanity_patterns = sn.all([
            sn.assert_ne(nids, []),
            sn.assert_ne(nids[0], nids[1])
        ])
        self.reference = {
            'daint:gpu': {
                'slope': (2.0, -0.1, 0.1, None),
                'offset': (0.0, -0.1, 0.1, None),
                'retries': (0, None, None, None),
                'time': (10, None, None, 's'),
            },
            'dom:gpu': {
                'slope': (2.0, -0.1, 0.1, None),
                'offset': (0.0, -0.1, 0.1, None),
                'retries': (0, None, None, None),
                'time': (10, None, None, 's'),
            }
        }
        self.perf_patterns = {
            'slope': sn.extractsingle(r'slope=(?P<slope>\S+)',
                                      self.stdout, 'slope', float),
            'offset': sn.extractsingle(r'offset=(?P<offset>\S+)',
                                       self.stdout, 'offset', float),
            'retries': 4 - sn.count(sn.findall(r'IPCluster is already running',
                                               self.stdout)),
            'time': sn.extractsingle(r'IPCluster is ready\!\s+'
                                     r'\((?P<time>\d+) seconds\)',
                                     self.stdout, 'time', float)
        }
        self.maintainers = ['RS', 'TR']
        self.tags = {'production'}

    @run_before('run')
    def prepare_run(self):
        # Change the job launcher since `ipython`
        # needs to be launched without `srun`.
        self.job.launcher = getlauncher('local')()
