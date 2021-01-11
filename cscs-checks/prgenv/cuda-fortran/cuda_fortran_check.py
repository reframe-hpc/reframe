# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
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

    @rfm.run_before('compile')
    def cdt_pgi_workaround(self):
        cdt = osext.cray_cdt_version()
        if not cdt:
            return

        if cdt == '20.08':
            self.build_system.fflags += [
                'CUDA_HOME=$CUDATOOLKIT_HOME', '-Mcuda=cuda10.2'
            ]
        else:
            #FIXME: workaround when CUDA 11.0 is the default version
            self.modules += ['cudatoolkit/10.2.89_3.29-7.0.2.1_3.5__g67354b4']
