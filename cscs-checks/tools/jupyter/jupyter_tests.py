# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
        self.sanity_patterns = sn.assert_found(r'nid\d+', self.stdout)
        self.tags = {'production', 'maintenance'}
        self.maintainers = ['RS', 'TR']
