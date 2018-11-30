import os
import re

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.required_version('>=2.16-dev0')
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

        self.sourcesdir = os.path.join(
            self.current_system.resourcesdir, 'CUDA', 'essentials'
        )
        self.build_system = 'SingleSource'
        self.sourcepath = 'bandwidthtestflex.cu'
        self.executable = 'gpu_bandwidth_check.x'

        # NOTE: Perform a range of bandwidth tests from 2MB to 32MB
        # with 2MB increments to avoid initialization overhead in bandwidth
        self.executable_opts = ['device', 'all', '--mode=range',
                                '--start=2097152', '--increment=2097152',
                                '--end=33554432', '--csv']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self.num_gpus_per_node = 1
        else:
            self.modules = ['craype-accel-nvidia35']
            self.num_gpus_per_node = 8

        self.sanity_patterns = self.do_sanity_check()
        self.reference = {}
        for d in range(self.num_gpus_per_node):
            self.reference['kesch:cn:bw_h2d_%i' % d] = (7213, -0.1, None, 'MB/s')
            self.reference['kesch:cn:bw_d2h_%i' % d] = (7213, -0.1, None, 'MB/s')
            self.reference['kesch:cn:bw_d2d_%i' % d] = (137347, -0.1, None, 'MB/s')
            self.reference['dom:gpu:bw_h2d_%i' % d] = (11648, -0.1, None, 'MB/s')
            self.reference['dom:gpu:bw_d2h_%i' % d] = (12423, -0.1, None, 'MB/s')
            self.reference['dom:gpu:bw_d2d_%i' % d] = (373803, -0.1, None, 'MB/s')
            self.reference['daint:gpu:bw_h2d_%i' % d] = (11648, -0.1, None, 'MB/s')
            self.reference['daint:gpu:bw_d2h_%i' % d] = (12423, -0.1, None, 'MB/s')
            self.reference['daint:gpu:bw_d2d_%i' % d] = (305000, -0.1, None, 'MB/s')

        # Set nvcc flags
        nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            nvidia_sm = '37'

        self.build_system.cxxflags = ['-I.', '-m64', '-arch=sm_%s' % nvidia_sm]
        self.maintainers = ['AJ', 'VK']
        self.tags = {'diagnostic', 'mch'}
        self.num_tasks_per_node = 1
        self.num_tasks = 0


    def _xfer_pattern(self, xfer_kind, devno, nodename):
        """generates search pattern for performance analysis"""
        if xfer_kind == 'h2d':
            first_part = 'bandwidthTest-H2D-Pinned'
        elif xfer_kind == 'd2h':
            first_part = 'bandwidthTest-D2H-Pinned'
        else:
            first_part = 'bandwidthTest-D2D'

        # Extract the bandwidth corresponding to the 32MB message (16th value)
        return (r'^%s[^,]*,\s*%s[^,]*,\s*Bandwidth\s*=\s*(\S+)\s*MB/s([^,]*,)'
                r'{2}\s*Size\s*=\s*33554432\s*bytes[^,]*,\s*DeviceNo\s*=\s*-1'
                r':%i' % (nodename, first_part, devno))


    @sn.sanity_function
    def do_sanity_check(self):
        failures = []
        all_detected_devices = set(sn.extractall(
            r'^\s*([^,]*),\s*Detected devices: %i' % self.num_gpus_per_node,
            self.stdout, 1
        ))
        number_of_detected_devices = len(all_detected_devices)

        if number_of_detected_devices != self.job.num_tasks:
            failures.append('Requested %s nodes, but found %s nodes)' %
                            (self.job.num_tasks, number_of_detected_devices))
            failures.append('nodelist %s' % all_detected_devices)
            sn.assert_false(failures, msg=', '.join(failures))

        all_tested_nodes_pass = set(sn.extractall(
            r'^\s*([^,]*),\s*NID\s*=\s*\S+\s+Result = PASS',
            self.stdout, 1
        ))
        if all_detected_devices != all_tested_nodes_pass:
            failures.append('nodes %s did not pass' %
                (all_detected_devices - all_tested_nodes_pass)
            )
            sn.assert_false(failures, msg=', '.join(failures))

        for nodename in all_detected_devices:
            self.perf_patterns = {}
            for xfer_kind in ['h2d', 'd2h', 'd2d']:
                for devno in range(self.num_gpus_per_node):
                    self.perf_patterns['bw_%s_%i' % (xfer_kind, devno)] = \
                        sn.extractsingle(self._xfer_pattern(xfer_kind, devno,
                            nodename), self.stdout, 1, float, 0
                        )

        return sn.assert_false(failures, msg=', '.join(failures))
