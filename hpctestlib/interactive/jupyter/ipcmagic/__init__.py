# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.backends import getlauncher


@rfm.simple_test
class ipcmagic_check(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Test ipcmagic via a distributed TensorFlow training with ipyparallel.

    `ipcmagic <https://github.com/eth-cscs/ipcluster_magic>`__ is a Python
    package and collection of CLI scripts for controlling clusters for
    Jupyter. For more information, please have a look
    `here <https://user.cscs.ch/tools/interactive/jupyterlab/>`__.

    This test checks the ipcmagic performance.
    To do this, a single-layer neural network is trained against a noisy linear
    function. The parameters of the fitted linear function are returned in the
    end along with the resulting loss function. The default assumption is that
    ipcmagic is already installed on the system under test.

    '''

    executable = 'ipython'
    executable_opts = ['--colors=NoColor', 'tf-hvd-sgd-ipc-tf2.ipynb']
    num_tasks = 2
    num_tasks_per_node = 1
    descr = 'Distributed training with TensorFlow using ipyparallel'

    @performance_function('N/A')
    def fitted_line_slope(self):
        return sn.extractsingle(r'slope=(?P<slope>\S+)',
                                self.stdout, 'slope', float)

    @performance_function('N/A')
    def fitted_line_offset(self):
        return sn.extractsingle(r'offset=(?P<offset>\S+)',
                                self.stdout, 'offset', float)

    @performance_function('N/A')
    def retries(self):
        return 4 - sn.count(sn.findall(r'IPCluster is already running',
                                       self.stdout))

    @run_before('run')
    def reset_launcher(self):
        # Change the job launcher since `ipython`
        # needs to be launched without `srun`.
        self.job.launcher = getlauncher('local')()

    @sanity_function
    def assert_successful_execution(self):
        '''Checks that the program is running on 2 different nodes (hostnames
        are different), that IPCMagic is configured and returns the correct
        end-of-program message (returns the slope parameter in the end).'''

        nids = sn.extractall(r"Out\[\d:1\]: '(?P<node>\S+)'", self.stdout,
                             'node', str)
        return sn.all([
            sn.assert_eq(sn.len(nids), 2),
            sn.assert_ne(nids[0], nids[1]),
            sn.assert_found(r'slope=\S+', self.stdout)
        ])
