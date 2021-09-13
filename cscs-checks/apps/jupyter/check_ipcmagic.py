# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
from reframe.core.backends import getlauncher

from hpctestlib.apps.jupyter.base_check import IPCMagic_BaseCheck

REFERENCE_PERFORMANCE = {
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

@rfm.simple_test
class IPCMagicCheck(IPCMagic_BaseCheck):
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    modules.append(
            f'Horovod/0.21.0-CrayGNU-{osext.cray_cdt_version()}-tf-2.4.0')
    num_tasks = 2
    num_tasks_per_node = 1
    maintainers = ['RS', 'TR']
    tags = {'production'}
    reference = REFERENCE_PERFORMANCE

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
