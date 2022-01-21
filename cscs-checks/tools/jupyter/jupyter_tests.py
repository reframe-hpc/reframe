# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class jupyterhub_submit_test(rfm.RunOnlyRegressionTest):
    valid_systems = ['daint:jupyter_gpu', 'daint:jupyter_mc',
                     'dom:jupyter_gpu', 'dom:jupyter_mc',
                     'eiger:jupyter_mc']
    valid_prog_environs = ['builtin']
    sourcesdir = None
    executable = 'hostname'
    time_limit = '1m'
    max_pending_time = '7m'
    tags = {'production', 'post-maintenance', 'health'}
    maintainers = ['RS', 'TR']

    @run_before('sanity')
    def set_sanity_check(self):
        self.sanity_patterns = sn.assert_found(r'nid\d+', self.stdout)


@rfm.simple_test
class jupyterhub_api_test(rfm.RunOnlyRegressionTest):
    descr = 'Check JupyterHub server status and version'
    valid_systems = ['daint:jupyter_gpu', 'daint:jupyter_mc',
                     'dom:jupyter_gpu', 'dom:jupyter_mc',
                     'eiger:jupyter_mc']
    valid_prog_environs = ['builtin']
    sourcesdir = None
    executable = 'curl https://jupyter.cscs.ch/hub/api/'
    time_limit = '30s'
    tags = {'health'}
    maintainers = ['CB', 'TR']

    @run_before('sanity')
    def set_sanity_check(self):
        self.sanity_patterns = sn.assert_found(r'{"version": "1.3.0"}',
                                               self.stdout)
