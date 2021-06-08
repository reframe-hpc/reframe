# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from hpctestlib.microbenchmarks.cpu.dgemm import Dgemm

@rfm.simple_test
class dgemm_check(Dgemm):
    valid_systems = ['dom:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    num_tasks = 0
    num_cpus_per_task = 36

    @run_before('compile')
    def setflags(self):
        self.build_system.cflags += ['-fopenmp']
