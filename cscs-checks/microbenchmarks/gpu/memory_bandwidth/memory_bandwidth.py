# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.simple_test
class GpuBandwidthCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100',
                              'ault:amda100', 'ault:amdvega']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']

        # Perform a single bandwidth test with a buffer size of 1024MB
        self.copy_size = 1073741824

        self.build_system = 'Make'
        self.executable = 'memory_bandwidth.x'
        self.build_system.cxxflags = [f'-DCOPY={self.copy_size}']
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.exclusive_access = True

        # perf_patterns and reference will be set by the sanity check function
        self.sanity_patterns = self.do_sanity_check()
        self.perf_patterns = {
            'h2d': sn.min(sn.extractall(self._xfer_pattern('h2d'),
                                        self.stdout, 1, float)),
            'd2h': sn.min(sn.extractall(self._xfer_pattern('d2h'),
                                        self.stdout, 1, float)),
            'd2d': sn.min(sn.extractall(self._xfer_pattern('d2d'),
                                        self.stdout, 1, float)),
        }
        self.reference = {
            'daint:gpu': {
                'h2d': (11881, -0.1, None, 'MB/s'),
                'd2h': (12571, -0.1, None, 'MB/s'),
                'd2d': (499000, -0.1, None, 'MB/s')
            },
            'dom:gpu': {
                'h2d': (11881, -0.1, None, 'MB/s'),
                'd2h': (12571, -0.1, None, 'MB/s'),
                'd2d': (499000, -0.1, None, 'MB/s')
            },
            'tsa:cn': {
                'h2d': (13000, -0.1, None, 'MB/s'),
                'd2h': (12416, -0.1, None, 'MB/s'),
                'd2d': (777000, -0.1, None, 'MB/s')
            },
            'ault:amda100': {
                'h2d': (25500, -0.1, None, 'MB/s'),
                'd2h': (26170, -0.1, None, 'MB/s'),
                'd2d': (1322500, -0.1, None, 'MB/s')
            },
            'ault:amdv100': {
                'h2d': (13189, -0.1, None, 'MB/s'),
                'd2h': (13141, -0.1, None, 'MB/s'),
                'd2d': (777788, -0.1, None, 'MB/s')
            },
            'ault:intelv100': {
                'h2d': (13183, -0.1, None, 'MB/s'),
                'd2h': (13411, -0.1, None, 'MB/s'),
                'd2d': (778200, -0.1, None, 'MB/s')
            },
            'ault:amdvega': {
                'h2d': (14000, -0.1, None, 'MB/s'),
                'd2h': (14000, -0.1, None, 'MB/s'),
                'd2d': (575700, -0.1, None, 'MB/s')
            },
        }
        self.tags = {'diagnostic', 'benchmark', 'mch',
                     'craype', 'external-resources'}
        self.maintainers = ['AJ', 'SK']

    @rfm.run_after('setup')
    def select_makefile(self):
        cp = self.current_partition.fullname
        if cp == 'ault:amdvega':
            self.build_system.makefile = 'makefile_memoryBandwidth.hip'
        else:
            self.build_system.makefile = 'makefile_memoryBandwidth.cuda'

    @rfm.run_after('setup')
    def set_gpu_arch(self):
        cp = self.current_partition.fullname

        # Deal with the NVIDIA options first
        nvidia_sm = None
        if cp in {'tsa:cn', 'ault:intelv100', 'ault:amdv100'}:
            nvidia_sm = '70'
        elif cp == 'ault:amda100':
            nvidia_sm = '80'
        elif cp in {'dom:gpu', 'daint:gpu'}:
            nvidia_sm = '60'

        if nvidia_sm:
            self.build_system.cxxflags += [f'-arch=sm_{nvidia_sm}']
            if cp in {'dom:gpu', 'daint:gpu'}:
                self.modules += ['craype-accel-nvidia60']
                if cp == 'dom:gpu':
                    self.modules += ['cdt-cuda']

            else:
                self.modules += ['cuda']

        # Deal with the AMD options
        amd_trgt = None
        if cp == 'ault:amdvega':
            amd_trgt = 'gfx906'

        if amd_trgt:
            self.build_system.cxxflags += [f'--amdgpu-target={amd_trgt}']
            self.modules += ['rocm']

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        cp = self.current_partition.fullname
        cs = self.current_system.name
        if cs in {'dom', 'daint'}:
            self.num_gpus_per_node = 1
        elif cs in {'arola', 'tsa'}:
            self.num_gpus_per_node = 8
        elif cp in {'ault:amda100', 'ault:intelv100'}:
            self.num_gpus_per_node = 4
        elif cp in {'ault:amdav100'}:
            self.num_gpus_per_node = 2
        elif cp in {'ault:amdvega'}:
            self.num_gpus_per_node = 3

    def _xfer_pattern(self, xfer_kind):
        '''generates search pattern for performance analysis'''
        if xfer_kind == 'h2d':
            direction = 'Host to device'
        elif xfer_kind == 'd2h':
            direction = 'Device to host'
        else:
            direction = 'Device to device'

        # Extract the bandwidth corresponding to the right node, transfer and
        # device.
        return (rf'^[^,]*\[[^,]*\]\s*{direction}\s*bandwidth on device'
                r' \d+ is \s*(\S+)\s*Mb/s.')

    @sn.sanity_function
    def do_sanity_check(self):
        node_names = set(sn.extractall(
            r'^\s*\[([^,]{1,20})\]\s*Found %s device\(s\).'
            % self.num_gpus_per_node, self.stdout, 1
        ))
        sn.evaluate(sn.assert_eq(
            self.job.num_tasks, len(node_names),
            msg='requested {0} node(s), got {1} (nodelist: %s)' %
            ','.join(sorted(node_names))))
        good_nodes = set(sn.extractall(
            r'^\s*\[([^,]{1,20})\]\s*Test Result\s*=\s*PASS',
            self.stdout, 1
        ))
        sn.evaluate(sn.assert_eq(
            node_names, good_nodes,
            msg='check failed on the following node(s): %s' %
            ','.join(sorted(node_names - good_nodes)))
        )

        return True
