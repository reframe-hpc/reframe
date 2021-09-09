# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class IPCMagic_BaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):

    #: :default: :class:`required`
    executable = required

    executable = 'ipython'
    executable_opts = ['tf-hvd-sgd-ipc-tf2.py']
    descr = 'Distributed training with TensorFlow using ipyparallel'
    '''
    @run_before('performance')
    def set_perf_patterns(self):
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
    '''
    @performance_function('None', perf_key='slope')
    def set_perf_slope(self):
        return sn.extractsingle(r'slope=(?P<slope>\S+)',
                                  self.stdout, 'slope', float)

    @performance_function('None', perf_key='offset')
    def set_perf_offset(self):
        return sn.extractsingle(r'offset=(?P<offset>\S+)',
                                   self.stdout, 'offset', float)

    @performance_function('None', perf_key='retries')
    def set_perf_retries(self):
        return 4 - sn.count(sn.findall(r'IPCluster is already running',
                                           self.stdout))

    @performance_function('s', perf_key='time')
    def set_perf_time(self):
        return sn.extractsingle(r'IPCluster is ready\!\s+'
                                r'\((?P<time>\d+) seconds\)',
                                self.stdout, 'time', float)

    @run_before('run')
    def prepare_run(self):
        # Change the job launcher since `ipython`
        # needs to be launched without `srun`.
        self.job.launcher = getlauncher('local')()

    @sanity_function
    def assert_energy_readout(self):
        '''Checks that the program is running on 2 different nodes (nids
        are different).'''

        nids = sn.extractall(r'nid(?P<nid>\d+)',
                             self.stdout, 'nid', str)
        return sn.all([
            sn.assert_ne(nids, []),
            sn.assert_ne(nids[0], nids[1])
        ])
