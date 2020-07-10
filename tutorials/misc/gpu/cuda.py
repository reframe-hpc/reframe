# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CUDATest(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'Matrix-vector multiplication example with CUDA'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['cray', 'gnu', 'pgi']
        self.sourcepath = 'matvec.cu'
        self.executable_opts = ['1024', '100']
        self.modules = ['cudatoolkit']
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout
        )
