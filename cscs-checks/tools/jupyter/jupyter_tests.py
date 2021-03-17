# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class JupyterHubSubmitTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:jupyter_gpu', 'daint:jupyter_mc',
                              'dom:jupyter_gpu', 'dom:jupyter_mc']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'hostname'
        self.time_limit = '1m'
        self.max_pending_time = '7m'
        self.sanity_patterns = sn.assert_found(r'nid\d+', self.stdout)
        self.tags = {'production', 'maintenance', 'health'}
        self.maintainers = ['RS', 'TR']


@rfm.simple_test
class JupyterHubServerCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Check JupyterHub server status and version'
        self.valid_systems = ['daint:jupyter_gpu', 'daint:jupyter_mc',
                              'dom:jupyter_gpu', 'dom:jupyter_mc']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'curl https://jupyter.cscs.ch/hub/api/'
        self.time_limit = '30s'
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_found(r'{"version": "1.3.0"}',
                                               self.stdout)
        self.tags = {'health'}
        self.maintainers = ['CB']
