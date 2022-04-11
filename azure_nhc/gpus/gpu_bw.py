# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# rfmdocstart: streamtest3
import os
import reframe as rfm
import reframe.utility.sanity as sn

# rfmdocstart: build_gpu_bandwidth
class build_gpu_bandwidth(rfm.CompileOnlyRegressionTest):
    descr = 'Build cuda bandwidth'
    build_system = 'Make'
    build_prefix = variable(str)
    valid_systems = ['*:gpu']

    @run_before('compile')
    def prepare_build(self):
        # Get node_data
        vm_info = self.current_system.node_data

        # Ideas for refactoring:
        # - get each name from ibstat -l and then check each one with ibstatus
        #   to see if the link_layer is InfiniBand or Ethernet (AccelNet)
        if vm_info != None and 'nhc_values' in vm_info:
            # Check if node has gpu
            if 'gpu_count' in vm_info['nhc_values']:
                if int(vm_info['nhc_values']['gpu_count']) <= 0:
                    return False
            else:
                return False
        fullpath = "/usr/local/cuda/samples/1_Utilities/bandwidthTest"
        current_line = "-I../../common/inc"
        updated_line = "-I/usr/local/cuda/samples/common/inc"
        self.prebuild_cmds = [
            f'cp -r {fullpath}/* {self.stagedir}/.',
            f'dir_name=`basename {fullpath}`',
            f'cd {self.stagedir}',
        ]
        #    f'sed -i "s/{current_line}/{updated_line}/g" Makefile'
        self.build_system.max_concurrency = 8

    @sanity_function
    def validate_build(self):
        # If compilation fails, the test would fail in any case, so nothing to
        # further validate here.
        return True
# rfmdocend: build_gpu_bandwidth

# rfmdocstart: gpu_bw
class CudaBandwidthTestBase(rfm.RunOnlyRegressionTest):
    '''Base class for Cuda bandwidth benchmark runtime test'''
    #valid_systems = ['*']
    valid_prog_environs = ['*']
    valid_systems = ['ndasr_v4', 'ndamsr_a100_v4']
    #valid_prog_environs = ['gnu']
    # rfmdocstart: cuda_bandwidth_binary
    osu_binaries = fixture(build_gpu_bandwidth, scope='environment')
    # rfmdocend: cuda_bandwidth_binary

    @sanity_function
    def validate_test(self):
        return sn.assert_found(r'^Running on...', self.stdout)

@rfm.simple_test
class gpu_bw_dtoh(CudaBandwidthTestBase):
    descr = 'GPU device to host bandwidth test'

    @run_before('run')
    def prepare_run(self):
        vm_info = self.current_system.node_data
        #print("vm_info: {}".format(vm_info))
        self.reference = {
            vm_info['vm_series']: {
                'Triad': (
                    vm_info['nhc_values']['stream_triad'],
                    vm_info['nhc_values']['stream_triad_limits'][0],
                    vm_info['nhc_values']['stream_triad_limits'][1],
                    'MB/s'
                )
            }
        }
        self.executable = "for device in {0..7}; do CUDA_VISIBLE_DEVICES=$device numactl -N$(( device / 2 )) -m$(( device / 2 )) ./bandwidthTest --dtoh ; done"

    @performance_function('GB/s')
    def bandwidth(self):
        bw_numbers = sn.extractall(r'32000000\s+(\S+)', self.stdout, 1, float)
        return sn.all([
            sn.assert_eq(sn.count(bw_numbers), 8),
            sn.all(sn.map(lambda x: sn.assert_bounded(x, 19, 27), bw_numbers))
        ])

@rfm.simple_test
class gpu_bw_htod(CudaBandwidthTestBase):
    descr = 'GPU host to device bandwidth test'

    @run_before('run')
    def prepare_run(self):
        self.executable = "for device in {0..7}; do CUDA_VISIBLE_DEVICES=$device numactl -N$(( device / 2 )) -m$(( device / 2 )) ./bandwidthTest --htod ; done"

    @performance_function('GB/s')
    def bandwidth(self):
        bw_numbers = sn.extractall(r'32000000\s+(\S+)', self.stdout, 1, float)
        return sn.all([
            sn.assert_eq(sn.count(bw_numbers), 8),
            sn.all(sn.map(lambda x: sn.assert_bounded(x, 19, 27), bw_numbers))
        ])
# rfmdocend: gpu_bw.py 
