# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class ContainerBase(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Test that asserts the ubuntu version of the image.'''

    # Derived tests must override this parameter
    platform = parameter()
    image_prefix = variable(str, value='')

    # Parametrize the test on two different versions of ubuntu.
    dist = parameter(['18.04', '20.04'])
    dist_name = variable(dict, value={
        '18.04': 'Bionic Beaver',
        '20.04': 'Focal Fossa',
    })

    @run_after('setup')
    def set_description(self):
        self.descr = (
            f'Run commands inside a container using ubuntu {self.dist}'
        )

    @run_before('run')
    def set_container_platform(self):
        self.container_platform = self.platform
        self.container_platform.image = (
            f'{self.image_prefix}ubuntu:{self.dist}'
        )
        self.container_platform.command = (
            "bash -c /rfm_workdir/get_os_release.sh"
        )

    @property
    def os_release_pattern(self):
        name = self.dist_name[self.dist]
        return rf'{self.dist}.\d+ LTS \({name}\)'

    @sanity_function
    def assert_release(self):
        return sn.all([
            sn.assert_found(self.os_release_pattern, 'release.txt'),
            sn.assert_found(self.os_release_pattern, self.stdout)
        ])
