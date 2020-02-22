# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NvmlCheck(rfm.RegressionTest):
    ''' This test checks gpu modes with nvml:
    * COMPUTE MODE:
    result = nvmlDeviceGetComputeMode(device, &compute_mode);

    * GPU OPERATION MODE (not supported since K20s, keeping as reminder):
    result = nvmlDeviceGetGpuOperationMode(device, &gom_mode_current,
                                                   &gom_mode_pending);
    NVML_GOM_ALL_ON Everything is enabled and running at full speed.
    NVML_GOM_COMPUTE Designed for running only compute tasks.
                     Graphics operations < are not allowed.
    NVML_GOM_LOW_DP Designed for running graphics applications that do not
                    require < high bandwidth double precision.
    '''

    def __init__(self):
        self.descr = 'check GPU compute mode'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'tiger:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['craype-accel-nvidia60']
        self.build_system = 'SingleSource'
        self.sourcepath = 'example.c'
        self.prebuild_cmd = [
            'cp $CUDATOOLKIT_HOME/nvml/example/example.c .',
            'patch -i ./nvml_example.patch'
        ]
        self.build_system.ldflags = ['-lnvidia-ml']
        if self.current_system.name in {'dom', 'daint'}:
            regex = (r"\s+Changing device.s compute mode from "
                     r"'Exclusive Process' to ")
        else:
            regex = r"\s+Changing device.s compute mode from 'Default' to "

        self.sanity_patterns = sn.assert_found(regex, self.stdout)
        self.maintainers = ['AJ', 'SK']
        self.tags = {'production', 'craype', 'external-resources'}
