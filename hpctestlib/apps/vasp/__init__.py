# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["VaspBaseCheck"]


class VASPBaseCheck(rfm.RunOnlyRegressionTest):
    num_tasks_per_node = required
    reference_value = variable(float)
    reference_difference = variable(float)
    keep_files = ['OUTCAR']

    @sanity_function
    def set_sanity_patterns(self):
        force = sn.extractsingle(r'1 F=\s+(?P<result>\S+)',
                                 self.stdout, 'result', float)
        force_diff = sn.abs(force - self.reference_value)
        ref_force_diff = sn.abs(self.reference_value *
                                self.reference_difference)

        return sn.assert_lt(force_diff, ref_force_diff)
