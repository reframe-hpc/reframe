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
        self.valid_prog_environs = ['PrgEnv-pgi', 'PrgEnv-nvidia']
        self.sourcepath = 'vecAdd_cuda.cuf'
        self.build_system = 'SingleSource'
        self.build_system.fflags = ['-ta=tesla:cc60', '-lcublas', '-lcusparse']
        self.num_gpus_per_node = 1
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
        self.maintainers = ['TM', 'AJ']
        self.tags = {'production', 'craype'}

    @run_before('compile')
    def pgi_20x_and_prgenv_nvidia_workaround(self):
        if self.current_system.name in ['daint']:
            self.modules += [f'cudatoolkit/{cudatoolkit_version}']
            if self.current_environ.name.startswith('PrgEnv-nvidia'):
                self.skip('PrgEnv-nvidia not supported on Daint')
        elif self.current_system.name in ['dom']:
            self.modules += [f'cdt-cuda/21.05']
            if self.current_environ.name.startswith('PrgEnv-pgi'):
                self.skip('PrgEnv-pgi not supported on Dom')
