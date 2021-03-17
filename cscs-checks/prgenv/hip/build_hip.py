import reframe as rfm
import reframe.utility.sanity as sn

import os


@rfm.simple_test
class BuildHip(rfm.RegressionTest):
    '''Download and install HIP around the nvcc compiler.'''

    # HIP build variables
    hip_path = variable(str, value='hip')
    hip_platform = variable(str, value='nvcc')

    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    sourcesdir =  'https://github.com/ROCm-Developer-Tools/HIP.git'
    build_system = 'CMake'
    postbuild_cmds = ['make install']
    executable = f'{hip_path}/bin/hipcc'
    executable_opts = ['--version']
    maintainers = ['JO']

    @rfm.run_before('compile')
    def set_compile_options(self):
        self.hip_full_path = os.path.abspath(
            os.path.join(self.stagedir, self.hip_path)
        )
        self.build_system.builddir = 'build'
        self.build_system.config_opts = [
            f'-DCMAKE_INSTALL_PREFIX={self.hip_full_path}',
            f'-DHIP_PLATFORM={self.hip_platform}',
            f'-DHIP_PATH={self.hip_full_path}'
        ]

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'nvcc:\s+NVIDIA', self.stdout)
