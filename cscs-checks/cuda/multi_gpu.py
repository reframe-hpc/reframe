import os

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.required_version('>=2.14')
@rfm.simple_test
class GpuBandwidthCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['kesch:cn', 'daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs += ['PrgEnv-cray-nompi',
                                         'PrgEnv-gnu-nompi']

        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CUDA', 'essentials')
        self.build_system = 'SingleSource'
        self.sourcepath = 'bandwidthTest.cu'
        self.executable = 'gpu_bandwidth_check.x'

        # NOTE: Perform a range of bandwidth tests from 2MB to 32MB
        # with 2MB increments to avoid initialization overhead in bandwidth
        self.executable_opts = ['device', 'all', '--mode=range',
                                '--start=2097152', '--increment=2097152',
                                '--end=33554432']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self.num_gpus_per_node = 1
        else:
            self.modules = ['craype-accel-nvidia35']
            self.num_gpus_per_node = 8

        self.sanity_patterns = sn.all([
            sn.assert_found('Result = PASS', self.stdout),
            sn.assert_found('Detected devices: %i' % self.num_gpus_per_node,
                            self.stdout)
        ])

        self.perf_patterns = {}
        for xfer_kind in ['h2d', 'd2h', 'd2d']:
            for device in range(self.num_gpus_per_node):
                self.perf_patterns['perf_%s_%i' % (xfer_kind, device)] = \
                    sn.extractsingle(self._xfer_pattern(xfer_kind, device),
                                     self.stdout, 3, float, 0)

        self.reference = {}
        for d in range(self.num_gpus_per_node):
            self.reference['kesch:cn:perf_h2d_%i' % d] = (7213, -0.1, None)
            self.reference['kesch:cn:perf_d2h_%i' % d] = (7213, -0.1, None)
            self.reference['kesch:cn:perf_d2d_%i' % d] = (137347, -0.1, None)
            self.reference['dom:gpu:perf_h2d_%i' % d] = (11648, -0.1, None)
            self.reference['dom:gpu:perf_d2h_%i' % d] = (12423, -0.1, None)
            self.reference['dom:gpu:perf_d2d_%i' % d] = (373803, -0.1, None)
            self.reference['daint:gpu:perf_h2d_%i' % d] = (11648, -0.1, None)
            self.reference['daint:gpu:perf_d2h_%i' % d] = (12423, -0.1, None)
            self.reference['daint:gpu:perf_d2d_%i' % d] = (305000, -0.1, None)

        # Set nvcc flags
        nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            nvidia_sm = '37'

        self.build_system.cxxflags = ['-I.', '-m64', '-arch=sm_%s' % nvidia_sm]

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def _xfer_pattern(self, xfer_kind, devno):
        """generates search pattern for performance analysis"""
        if xfer_kind == 'h2d':
            first_part = 'Host to Device Bandwidth'
        elif xfer_kind == 'd2h':
            first_part = 'Device to Host Bandwidth'
        else:
            first_part = 'Device to Device Bandwidth'

        # Extract the bandwidth corresponding to the 32MB message (16th value)
        return (r'^ *%s([^\n]*\n){%i}^ *Device Id: %i\s+'
                r'([^\n]*\n){15}'
                r'\s+\d+\s+(\S+)' % (first_part, 3+18*devno, devno))
