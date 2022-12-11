# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check


def _hecbiosim_bench(params):
    for p in params:
        if p[0] == 'HECBioSim/hEGFRDimerSmallerPL':
            return [p]


@rfm.simple_test
class gromacs_containerized_test(gromacs_check):
    # Restrict library test parameters to only those relevant for this example
    benchmark_info = parameter(inherit_params=True,
                               filter_params=_hecbiosim_bench,
                               fmt=lambda x: x[0])
    nb_impl = parameter(['gpu'])

    # New parameter for testing the various images
    gromacs_image = parameter([
        None,
        'nvcr.io/hpc/gromacs:2020',
        'nvcr.io/hpc/gromacs:2020.2',
        'nvcr.io/hpc/gromacs:2021',
        'nvcr.io/hpc/gromacs:2021.3',
        'nvcr.io/hpc/gromacs:2022.1'
    ])
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['gnu']
    use_multithreading = False
    executable = 'gmx mdrun'
    executable_opts += ['-dlb yes', '-ntomp 12', '-npme -1', '-v']
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 12

    @run_after('init')
    def setup_container_run(self):
        exec_cmd = ' '.join([self.executable, *self.executable_opts])
        self.container_platform.image = self.gromacs_image
        self.container_platform.command = exec_cmd
        if self.gromacs_image is None:
            self.modules = ['daint-gpu', 'GROMACS']
