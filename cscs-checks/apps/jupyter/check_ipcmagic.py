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
    descr = 'Distributed training with TensorFlow using ipyparallel'
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    modules = [
        f'ipcmagic', f'jupyterlab',
        f'Horovod/0.21.0-CrayGNU-{osext.cray_cdt_version()}-tf-2.4.0'
    ]
    num_tasks = 2
    num_tasks_per_node = 1
    executable = 'ipython'
    executable_opts = ['tf-hvd-sgd-ipc-tf2.py']
    reference = {
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

    maintainers = ['RS', 'TR']
    tags = {'production'}

    @sanity_function
    def assert_nids(self):
        nids = sn.extractall(r'nid(?P<nid>\d+)', self.stdout, 'nid', str)
        return sn.all([sn.assert_ne(nids, []), sn.assert_ne(nids[0], nids[1])])

    @performance_function('')
    def slope(self):
        return sn.extractsingle(r'slope=(?P<slope>\S+)', self.stdout,
                                'slope', float)

    @performance_function('')
    def offset(self):
        return sn.extractsingle(r'offset=(?P<offset>\S+)', self.stdout,
                                'offset', float)

    @performance_function('')
    def retries(self):
        return 4 - sn.count(sn.findall(r'IPCluster is already running',
                                       self.stdout))

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'IPCluster is ready\!\s+'
                                r'\((?P<time>\d+) seconds\)',
                                self.stdout, 'time', float)

    @run_before('run')
    def reset_launcher(self):
        # Change the job launcher since `ipython`
        # needs to be launched without `srun`.
        self.job.launcher = getlauncher('local')()
