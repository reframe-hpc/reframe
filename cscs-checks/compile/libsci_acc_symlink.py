# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class LibSciAccSymLinkTest(rfm.RunOnlyRegressionTest):
    lib_name = parameter([
        'libsci_acc_gnu_81_nv35', 'libsci_acc_gnu_81_nv60',
        'libsci_acc_cray_nv35', 'libsci_acc_cray_nv60',
        'libsci_acc_cray_nv35_openacc', 'libsci_acc_cray_nv60_openacc'
    ])

    def __init__(self):
        self.descr = f'LibSciAcc symlink check of {self.lib_name}'
        self.valid_systems = [
            'daint:login', 'daint:gpu',
            'dom:login', 'dom:gpu',
        ]
        regex = (r'libsci_acc_(?P<prgenv>[A-Za-z]+)_((?P<cver>[A-Za-z0-9]+)_)'
                 r'?(?P<version>\S+)')
        prgenv = re.match(regex, self.lib_name).group('prgenv')

        # The prgenv is irrelevant for this case, so just chose one
        self.valid_prog_environs = ['builtin']
        self.executable = 'ls'
        self.executable_opts = ['-al', '/opt/cray/pe/lib64/libsci_a*']
        self.sanity_patterns = sn.assert_found(f'{self.lib_name}.so',
                                               self.stdout)

        self.maintainers = ['AJ', 'LM']
        self.tags = {'production', 'craype', 'health'}
