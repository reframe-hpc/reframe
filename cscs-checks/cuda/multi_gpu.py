import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class GpuBandwidthCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('gpu_bandwidth_check',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['kesch:cn', 'daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']

        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CUDA', 'essentials')
        self.sourcepath = 'bandwidthTest.cu'
        self.executable = 'gpu_bandwidth_check.x'
        self.executable_opts = ['device', 'all']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['cudatoolkit']
            self.num_gpus_per_node = 1
        else:
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
                                     self.stdout, 2, float, 0)

        kesch_cn = {}
        daint_gpu = {}
        dom_gpu = {}
        for device in range(self.num_gpus_per_node):
            kesch_cn['perf_h2d_%i' % device] = (7213, -0.1, None)
            kesch_cn['perf_d2h_%i' % device] = (7213, -0.1, None)
            kesch_cn['perf_d2d_%i' % device] = (137347, -0.1, None)
            dom_gpu['perf_h2d_%i' % device] = (11648, -0.1, None)
            dom_gpu['perf_d2h_%i' % device] = (12423, -0.1, None)
            dom_gpu['perf_d2d_%i' % device] = (373803, -0.1, None)
            daint_gpu['perf_h2d_%i' % device] = (11648, -0.1, None)
            daint_gpu['perf_d2h_%i' % device] = (12423, -0.1, None)
            daint_gpu['perf_d2d_%i' % device] = (305000, -0.1, None)

        self.reference = {
            'kesch:cn': kesch_cn,
            'daint:gpu': daint_gpu,
            'dom:gpu': dom_gpu
        }

        # Set nvcc flags
        nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            nvidia_sm = '37'
        self._flags = ('-m64 -arch=sm_%s' % nvidia_sm)

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if self.current_system.name == 'kesch' and environ.name == 'PrgEnv-gnu':
            self.modules = ['craype-accel-nvidia35']

        super().setup(partition, environ, **job_opts)

    def compile(self):
        self.current_environ.cxxflags = self._flags
        super().compile()

    def _xfer_pattern(self, xfer_kind, devno):
        """generates search pattern for performance analysis"""
        if xfer_kind == 'h2d':
            first_part = 'Host to Device Bandwidth'
        elif xfer_kind == 'd2h':
            first_part = 'Device to Host Bandwidth'
        else:
            first_part = 'Device to Device Bandwidth'
        return r'^ *%s([^\n]*\n){%i}^ *Device Id: %i[^\n]*\n^\s*\d+\s+(\S+)' % \
               (first_part, 3+3*devno, devno)


def _get_checks(**kwargs):
    return [GpuBandwidthCheck(**kwargs)]
