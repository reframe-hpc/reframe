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
            'slope': (2.0, -0.1, 0.1, 'N/A'),
            'offset': (0.0, -0.1, 0.1, 'N/A'),
            'retries': (0, None, None, 'N/A'),
            'time': (10, None, None, 's'),
        },
        'dom:gpu': {
            'slope': (2.0, -0.1, 0.1, 'N/A'),
            'offset': (0.0, -0.1, 0.1, 'N/A'),
            'retries': (0, None, None, 'N/A'),
            'time': (10, None, None, 's'),
        }
    }

    maintainers = ['RS', 'TR']
    tags = {'production'}

    @run_after('setup')
    def daint_module_workaround(self):
        if self.current_system.name == 'daint':
            # FIXME: Use the default modules once Dom/Daint are aligned
            self.modules = [
                f'ipcmagic/1.0.1-CrayGNU-{osext.cray_cdt_version()}',
                f'Horovod/0.21.0-CrayGNU-{osext.cray_cdt_version()}-tf-2.4.0'
            ]
            # FIXME: Enforce loading of jupyterlab module since
            # `module show jupyterlab` throws a Tcl error on Daint
            self.prerun_cmds = ['module load jupyterlab']

    @sanity_function
    def assert_successful_execution(self):
        nids = sn.extractall(r'nid(?P<nid>\d+)', self.stdout, 'nid', str)
        return sn.all([
            sn.assert_ne(nids, []), sn.assert_ne(nids[0], nids[1]),
            sn.assert_found(r'IPCluster is ready\!\s+', self.stdout),
            sn.assert_found(r'slope=\S+', self.stdout)
        ])

    @performance_function('N/A')
    def slope(self):
        return sn.extractsingle(r'slope=(?P<slope>\S+)', self.stdout,
                                'slope', float)

    @performance_function('N/A')
    def offset(self):
        return sn.extractsingle(r'offset=(?P<offset>\S+)', self.stdout,
                                'offset', float)

    @performance_function('N/A')
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
