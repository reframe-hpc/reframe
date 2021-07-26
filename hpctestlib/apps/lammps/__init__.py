# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["LammpsBaseCheck"]


class LAMMPSBaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    input_file = variable(str)
    num_tasks_per_node = required
    energy_reference = variable(float)
    energy_difference = variable(float)

    @sanity_function
    def set_sanity_patterns(self):
        energy = sn.extractsingle(
            r'\s+500000(\s+\S+){3}\s+(?P<energy>\S+)\s+\S+\s\n',
            self.stdout, 'energy', float)
        energy_diff = sn.abs(energy - self.energy_reference)
        ref_ener_diff = sn.abs(self.energy_reference *
                               self.energy_difference)

        return sn.all([
            sn.assert_found(r'Total wall time:', self.stdout),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
