# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.apps.jupyter.ipcmagic import IPCMagic

REFERENCE_PERFORMANCE = {
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

@rfm.simple_test
class ipc_magic_check(IPCMagic):
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['builtin']
    modules = ['ipcmagic', 'jupyterlab', 'Horovod']
    num_tasks = 2
    num_tasks_per_node = 1
    maintainers = ['RS', 'TR']
    tags = {'production'}
    reference = REFERENCE_PERFORMANCE
