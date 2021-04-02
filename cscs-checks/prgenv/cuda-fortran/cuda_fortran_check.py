# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CUDAFortranCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-pgi']
        self.sourcepath = 'vecAdd_cuda.cuf'
        self.modules = ['craype-accel-nvidia60']
        self.build_system = 'SingleSource'
        self.build_system.fflags = ['-ta=tesla:cc60']
        self.num_gpus_per_node = 1
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
        self.maintainers = ['TM', 'AJ']
        self.tags = {'production', 'craype'}

    # FIXME: PGI 20.x does not support CUDA 11, see case #275674
    @rfm.run_before('compile')
    def cudatoolkit_pgi_20x_workaround(self):
        if self.current_system.name == 'daint':
            cudatoolkit_version = '10.2.89_3.29-7.0.2.1_3.5__g67354b4'
        else:
            cudatoolkit_version = '10.2.89_3.29-7.0.2.1_3.27__g67354b4'

        self.modules += [f'cudatoolkit/{cudatoolkit_version}']
