# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CUDAFortranCheck(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-pgi', 'PrgEnv-nvidia']
    sourcepath = 'vecAdd_cuda.cuf'
    build_system = 'SingleSource'
    num_gpus_per_node = 1
    maintainers = ['TM', 'AJ']
    tags = {'production', 'craype'}

    @run_after('setup')
    def set_modules(self):
        if self.current_environ.name != 'PrgEnv-nvidia':
            self.modules = ['craype-accel-nvidia60']
        else:
            self.modules = ['cdt-cuda/21.05']

    @run_before('compile')
    def set_fflags(self):
        self.build_system.fflags = ['-ta=tesla:cc60']

    # FIXME: PGI 20.x does not support CUDA 11, see case #275674
    @run_before('compile')
    def cudatoolkit_pgi_20x_workaround(self):
        if self.current_system.name == 'daint':
            cudatoolkit_version = '10.2.89_3.29-7.0.2.1_3.27__g67354b4'
        else:
            self.variables['CUDA_HOME'] = '$CUDATOOLKIT_HOME'
            cudatoolkit_version = '10.2.89_3.28-2.1__g52c0314'

        self.modules += [f'cudatoolkit/{cudatoolkit_version}']

    @run_before('sanity')
    def set_sanity(self):
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)

