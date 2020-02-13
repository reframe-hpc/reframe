import os

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class GpuBandwidthCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['kesch:cn', 'daint:gpu', 'dom:gpu', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['arolla', 'kesch', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']
            self.exclusive_access = True

        self.sourcesdir = os.path.join(
            self.current_system.resourcesdir, 'CUDA', 'essentials'
        )
        self.build_system = 'SingleSource'

        # Set nvcc flags
        nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            nvidia_sm = '37'
        elif self.current_system.name in ['arolla', 'tsa']:
            nvidia_sm = '70'

        self.build_system.cxxflags = ['-I.', '-m64', '-arch=sm_%s' % nvidia_sm]
        self.sourcepath = 'bandwidthtestflex.cu'
        self.executable = 'gpu_bandwidth_check.x'

        # Perform a single bandwidth test with a buffer size of 1024MB
        self.min_buffer_size = 1073741824
        self.max_buffer_size = 1073741824
        self.executable_opts = ['device', 'all', '--mode=range',
                                '--start=%d' % self.min_buffer_size,
                                '--increment=%d' % self.min_buffer_size,
                                '--end=%d' % self.max_buffer_size, '--csv']
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        if self.current_system.name in ['daint', 'dom', 'tiger']:
            self.modules = ['craype-accel-nvidia60']
            self.num_gpus_per_node = 1
        elif self.current_system.name == 'kesch':
            self.modules = ['cudatoolkit/8.0.61']
            self.num_gpus_per_node = 8
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']
            self.num_gpus_per_node = 8

        # perf_patterns and reference will be set by the sanity check function
        self.sanity_patterns = self.do_sanity_check()
        self.perf_patterns = {}
        self.reference = {}
        self.__bwref = {
        # FIXME: reference values for Arolla and Tsa need to be updated 
        #    (sanity check fails if they are not defined)
            'arolla:cn:h2d':   (7583, -0.1, None, 'MB/s'),
            'arolla:cn:d2h':   (7584, -0.1, None, 'MB/s'),
            'arolla:cn:d2d': (137408, -0.1, None, 'MB/s'),
            'daint:gpu:h2d':  (11881, -0.1, None, 'MB/s'),
            'daint:gpu:d2h':  (12571, -0.1, None, 'MB/s'),
            'daint:gpu:d2d': (499000, -0.1, None, 'MB/s'),
            'dom:gpu:h2d':  (11881, -0.1, None, 'MB/s'),
            'dom:gpu:d2h':  (12571, -0.1, None, 'MB/s'),
            'dom:gpu:d2d': (499000, -0.1, None, 'MB/s'),
            'kesch:cn:h2d':   (7583, -0.1, None, 'MB/s'),
            'kesch:cn:d2h':   (7584, -0.1, None, 'MB/s'),
            'kesch:cn:d2d': (137408, -0.1, None, 'MB/s'),
            'tiger:gpu:h2d': (0, None, None, 'MB/s'),
            'tiger:gpu:d2h': (0, None, None, 'MB/s'),
            'tiger:gpu:d2d': (0, None, None, 'MB/s'),
            'tsa:cn:h2d':   (7583, -0.1, None, 'MB/s'),
            'tsa:cn:d2h':   (7584, -0.1, None, 'MB/s'),
            'tsa:cn:d2d': (137408, -0.1, None, 'MB/s'),
        }
        self.tags = {'diagnostic', 'benchmark', 'mch',
                     'craype', 'external-resources'}
        self.maintainers = ['AJ', 'SK']

    def _xfer_pattern(self, xfer_kind, devno, nodename):
        '''generates search pattern for performance analysis'''
        if xfer_kind == 'h2d':
            first_part = 'bandwidthTest-H2D-Pinned'
        elif xfer_kind == 'd2h':
            first_part = 'bandwidthTest-D2H-Pinned'
        else:
            first_part = 'bandwidthTest-D2D'

        # Extract the bandwidth corresponding to the maximum buffer size
        return (r'^%s[^,]*,\s*%s[^,]*,\s*Bandwidth\s*=\s*(\S+)\s*MB/s([^,]*,)'
                r'{2}\s*Size\s*=\s*%d\s*bytes[^,]*,\s*DeviceNo\s*=\s*-1'
                r':%s' % (nodename, first_part, self.max_buffer_size, devno))

    @sn.sanity_function
    def do_sanity_check(self):
        failures = []
        devices_found = set(sn.extractall(
            r'^\s*([^,]*),\s*Detected devices: %s' % self.num_gpus_per_node,
            self.stdout, 1
        ))

        sn.evaluate(sn.assert_eq(
            self.job.num_tasks, len(devices_found),
            msg='requested {0} node(s), got {1} (nodelist: %s)' %
            ','.join(sorted(devices_found))))

        good_nodes = set(sn.extractall(
            r'^\s*([^,]*),\s*NID\s*=\s*\S+\s+Result = PASS',
            self.stdout, 1
        ))

        sn.evaluate(sn.assert_eq(
            devices_found, good_nodes,
            msg='check failed on the following node(s): %s' %
            ','.join(sorted(devices_found - good_nodes)))
        )

        # Sanity is fine, fill in the perf. patterns based on the exact node id
        for nodename in devices_found:
            for xfer_kind in ('h2d', 'd2h', 'd2d'):
                for devno in range(self.num_gpus_per_node):
                    perfvar = '%s_gpu_%s_%s_bw' % (nodename, devno, xfer_kind)
                    perfvar = 'bw_%s_%s_gpu_%s' % (xfer_kind, nodename, devno)
                    self.perf_patterns[perfvar] = sn.extractsingle(
                        self._xfer_pattern(xfer_kind, devno, nodename),
                        self.stdout, 1, float, 0
                    )
                    partname = self.current_partition.fullname
                    refkey = '%s:%s' % (partname, perfvar)
                    bwkey = '%s:%s' % (partname, xfer_kind)
                    self.reference[refkey] = self.__bwref[bwkey]

        return True
