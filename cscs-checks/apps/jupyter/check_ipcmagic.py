# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.interactive.jupyter.ipcmagic import ipcmagic_check


@rfm.simple_test
class cscs_ipcmagic_check(ipcmagic_check):
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['builtin']
    modules = ['Horovod', 'jupyterlab']

    @run_after('init')
    def modules_workaround(self):
        if self.current_system.name == 'dom':
            self.modules = ['Horovod', 'JuliaExtensions', 'jupyterlab']

    maintainers = ['RS', 'TR']
    tags = {'production'}
    reference = {
        'daint:gpu': {
            'slope': (2.0, -0.1, 0.1, 'N/A'),
            'offset': (0.0, -0.1, 0.1, 'N/A'),
            'retries': (0, None, None, 'N/A'),
        },
        'dom:gpu': {
            'slope': (2.0, -0.1, 0.1, 'N/A'),
            'offset': (0.0, -0.1, 0.1, 'N/A'),
            'retries': (0, None, None, 'N/A'),
        }
    }
