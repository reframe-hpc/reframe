# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import tutorials.advanced.library.lib as lib


@rfm.simple_test
class ContainerTest(lib.ContainerBase):
    platform = parameter(['Sarus', 'Singularity'])
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']

    @run_after('setup')
    def set_image_prefix(self):
        if self.platform == 'Singularity':
            self.image_prefix = 'docker://'
