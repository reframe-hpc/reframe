# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.os_ext as os_ext
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class IPCMagicCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Distributed training with TensorFlow using ipyparallel'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        # FIXME: The following will not be needed after the Daint upgrade
        cray_cdt_version = os_ext.cray_cdt_version() or '19.10'
        self.modules = [
            'ipcmagic',
            f'Horovod/0.19.1-CrayGNU-{cray_cdt_version}-tf-2.2.0'
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

    @rfm.run_before('run')
    def prepare_run(self):
        # Change the job launcher since `ipython`
        # needs to be launched without `srun`.
        self.job.launcher = getlauncher('local')()
