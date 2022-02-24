# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# rfmdocstart: streamtest4
import reframe as rfm
import reframe.utility.sanity as sn
import inspect
import reframe.core.config as cfg
import pprint

@rfm.simple_test
class IBCardCheck(rfm.RunOnlyRegressionTest):
    descr = 'Check the number of IB cards'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'lspci'


    @sanity_function
    def validate_results(self):
        # Get node_data
        vm_info = self.current_system.node_data

        # Ideas for refactoring:
        # - get each name from ibstat -l and then check each one with ibstatus
        #   to see if the link_layer is InfiniBand or Ethernet (AccelNet)
        if vm_info != None and 'nhc_values' in vm_info:
            ib_devices= []
            nvme_devices = []
            gpu_devices = []
            an_devices = []
            if "ib_count" in vm_info['nhc_values'] and "pcie_ib_name" in vm_info['nhc_values']:
                ib_devices = sn.extractall(
                    r'({})'.format(vm_info["nhc_values"]["pcie_ib_name"]), self.stdout
                )
            if "nvme_count" in vm_info['nhc_values'] and "pcie_nvme_name" in vm_info['nhc_values']:
                nvme_devices = sn.extractall(
                    r'({})'.format(vm_info["nhc_values"]["pcie_nvme_name"]), self.stdout
                )
            if "gpu_count" in vm_info['nhc_values'] and "pcie_gpu_name" in vm_info['nhc_values']:
                gpu_devices = sn.extractall(
                    r'({})'.format(vm_info["nhc_values"]["pcie_gpu_name"]), self.stdout
                )
            if "pcie_accel_net_name" in vm_info['nhc_values']:
                an_devices = sn.extractall(
                    r'({})'.format(vm_info["nhc_values"]["pcie_accel_net_name"]), self.stdout
                )

            #print("PCIe IB Cards: {}".format(ib_devices))
            #print("Count: {}".format(sn.count(ib_devices)))
            #print("PCIe NVMe Disks: {}".format(nvme_devices))
            #print("Count: {}".format(sn.count(nvme_devices)))
            #print("PCIe GPU Devices: {}".format(gpu_devices))
            #print("Count: {}".format(sn.count(gpu_devices)))
            #print("PCIe Accel Net Cards: {}".format(an_devices))
        
            num_ib_count = 0
            num_an_count = 0
            num_gpus = 0
            num_nvme_count = 0
            if 'ib_count' in vm_info['nhc_values']:
                num_ib_count = vm_info['nhc_values']['ib_count']
            if 'nvme_count' in vm_info['nhc_values']:
                num_nvme_count = vm_info['nhc_values']['nvme_count']
            if 'gpu_count' in vm_info['nhc_values']:
                num_gpus = vm_info['nhc_values']['gpu_count']

            print("IB Devices : {} : expected {}".format(sn.count(ib_devices), num_ib_count))
            print("NVMe Disks : {} : expected {}".format(sn.count(nvme_devices), num_nvme_count))
            print("GPUs       : {} : expected {}".format(sn.count(gpu_devices), num_gpus))

            return sn.all([
                sn.assert_eq(sn.count(ib_devices), num_ib_count),
                sn.assert_eq(sn.count(nvme_devices), num_nvme_count),
                sn.assert_eq(sn.count(gpu_devices), num_gpus)
            ])    


