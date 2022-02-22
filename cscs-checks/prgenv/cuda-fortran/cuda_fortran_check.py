# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CUDAFortranCheck(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-nvidia']
    sourcepath = 'vecAdd_cuda.cuf'
    build_system = 'SingleSource'
    num_gpus_per_node = 1
    maintainers = ['TM', 'AJ']
    tags = {'production', 'craype'}

    @run_before('compile')
    def set_fflags(self):
        self.build_system.fflags = ['-ta=tesla:cc60']

    @run_before('sanity')
    def set_sanity(self):
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
