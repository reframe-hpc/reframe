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
    executable = 'ibstat -l'


    @sanity_function
    def validate_results(self):
        # Get node_data
        vm_info = self.current_system.node_data
        if 'runtime_data' not in self.current_system.node_data:
           self.current_system.node_data['runtime_data'] = {}
        self.current_system.node_data['runtime_data']['accelnet'] = True


        # Ideas for refactoring:
        # - get each name from ibstat -l and then check each one with ibstatus
        #   to see if the link_layer is InfiniBand or Ethernet (AccelNet)
        ib_count = sn.extractall(
            r'(mlx5_ib[0-9]+)', self.stdout
            #r'(?P<name>\S+)', self.stdout
        )
        print("IB Cards: {}".format(ib_count))
        print("Count: {}".format(sn.count(ib_count)))
        #return sn.assert_eq(sn.count(ib_count), 1 )
        #print("=====================")
        pprint.pprint(vars(self.current_system))
        #print("=========------------============")
        if vm_info != None and 'nhc_values' in vm_info and "ib_count" in vm_info['nhc_values']:
            return sn.assert_eq(sn.count(ib_count), vm_info['nhc_values']['ib_count'])
        else:
            print("ib_count not found in vm_info['nhc_values']")
            return sn.assert_eq(sn.count(ib_count), 0)


