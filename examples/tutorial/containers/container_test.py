# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ContainerTest(rfm.RunOnlyRegressionTest):
    container_platform = 'Docker'
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_before('run')
    def setup_container_platf(self):
        self.container_platform.image = f'ubuntu:18.04'
        self.container_platform.command = (
            "bash -c 'cat /etc/os-release | tee /rfm_workdir/release.txt'"
        )

    @sanity_function
    def assert_release(self):
        os_release_pattern = r'18.04.\d+ LTS \(Bionic Beaver\)'
        return sn.assert_found(os_release_pattern, 'release.txt')
