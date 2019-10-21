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
        super().__init__()
        self.descr = 'check GPU compute mode'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['craype-accel-nvidia60']
        self.build_system = 'SingleSource'
        self.sourcepath = 'example.c'
        self.prebuild_cmd = [
            'cp $CUDATOOLKIT_HOME/nvml/example/example.c .',
            'patch -i ./nvml_example.patch'
        ]
        self.build_system.ldflags = ['-lnvidia-ml']
        self.sanity_patterns = sn.assert_found(
            r"\s+Changing device.s compute mode from 'Exclusive Process' to ",
            self.stdout)

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production', 'craype', 'external-resources'}
