# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import time

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


@rfm.simple_test
class ssh_host_keys_check(rfm.RunOnlyRegressionTest):
    '''SSH host keys age check

    The test checks if the list of SSH keys has been updated recently.
    In this case, we are checking against the max_key_age variable
    '''

    #: Parameter list with all host keys to check
    #:
    #: The test is skipped if a key is not found
    #:
    #: :type: :class:`str`
    #: :values: ``['/etc/ssh/ssh_host_rsa_key',
    #:              '/etc/ssh/ssh_host_ecdsa_key',
    #:              '/etc/ssh/ssh_host_ed25519_key']``
    ssh_host_keys = parameter([
        '/etc/ssh/ssh_host_rsa_key',
        '/etc/ssh/ssh_host_ecdsa_key',
        '/etc/ssh/ssh_host_ed25519_key',
    ], fmt=lambda x: x.split('_')[2], loggable=True)

    #: The max age of the keys in ReFrame duration format
    #:
    #: :type: :class:`str`
    #: :default: ``'365d'``
    max_key_age = variable(str, value='365d', loggable=True)

    executable = 'stat'
    executable_opts = ['-c', '%Y']
    tags = {'system', 'ssh'}

    @run_after('init')
    def set_hosts_keys(self):
        self.executable_opts += [self.ssh_host_keys]

    @sanity_function
    def assert_file_age(self):
        current_time = time.time()

        skip_me = sn.extractall('No such file or directory', self.stderr)
        self.skip_if(skip_me, msg=f'Skipping test because {self.ssh_host_keys}'
                                  f' was not found')

        return sn.assert_lt(current_time -
                         sn.extractsingle(r'\d+', self.stdout, 0, int),
                         typ.Duration(self.max_key_age),
                         msg=f'File {self.ssh_host_keys} is older than '
                             f'{self.max_key_age}')
