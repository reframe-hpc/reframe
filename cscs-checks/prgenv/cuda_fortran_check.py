# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CUDAFortranCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'tiger:gpu']
        self.valid_prog_environs = ['PrgEnv-pgi']
        self.sourcepath = 'vecAdd_cuda.f90'
        self.modules = ['craype-accel-nvidia60']
        self.build_system = 'SingleSource'
        self.num_gpus_per_node = 1
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
        self.maintainers = ['TM', 'AJ']
        self.tags = {'production', 'craype'}

    @rfm.run_before('compile')
    def setflags(self):
        # FIXME CUDA_HOME should not be set this way
        self.build_system.fflags = [
            'CUDA_HOME=$CUDATOOLKIT_HOME',
            '-ta=tesla:cc60', '-Mcuda=cuda10.2'
        ]
