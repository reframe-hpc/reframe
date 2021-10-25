# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.backends import getlauncher


@rfm.simple_test
class ipcmagic_check(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Test IPCMagic via a distributed TensorFlow training with ipyparallel.

    MPI for Python provides bindings of the Message Passing Interface
    (MPI) standard for Python, allowing any Python program to exploit
    multiple processors.

    MPI can be made available on Jupyter notebooks through IPyParallel.
    This is a Python package and collection of CLI scripts for controlling
    clusters for Jupyter: A set of servers that act as a cluster, called
    engines, is created and the code in the notebook's cells will be executed
    within them. This cluster can be run within a single node, or spanning
    multiple nodes.

    The engines and another server that moderates the cluster, called the
    controller, can be started an stopped with the magic %ipcluster start n
    <num-engines> --mpi and %ipcluster stop, respectively.

    This test checks the IPCMagic performance.
    To do this, a single-layer neural network is trained against a noisy linear
    function. The parameters of the fitted linear function are returned in the
    end along with the resulting loss function. The default assumption is that
    IPCMagic is already installed on the system under test.
    '''

    executable = 'ipython'
    executable_opts = ['tf-hvd-sgd-ipc-tf2.py']
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
        '''Checks that the program is running on 2 different nodes (nids
        are different), that IPCMagic is configured and returns the correct
        end-of-program message (return the parameter slope in the end).'''

        nids = sn.extractall(r'nid(?P<nid>\d+)', self.stdout, 'nid', str)
        return sn.all([
            sn.assert_eq(sn.len(nids), 2),
            sn.assert_ne(nids[0], nids[1]),
            sn.assert_found(r'slope=\S+', self.stdout)
        ])