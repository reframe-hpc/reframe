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

        ib_cards = sn.extractall(
            r'(?P<name>\S+)', self.stdout
        )
        print("IB Cards: {}".format(ib_cards))
        print("Count: {}".format(sn.count(ib_cards)))
        #return sn.assert_eq(sn.count(ib_cards), 1 )
        return sn.assert_eq(sn.count(ib_cards), vm_info['nhc_values']['ib_cards'])

