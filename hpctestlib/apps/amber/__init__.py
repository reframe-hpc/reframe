#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["AmberBaseCheck"]


class AmberBaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    input_file = variable(str)
    output_file = variable(str)
    num_tasks_per_node = required
    reference_value = variable(float)
    reference_difference = variable(float)

    @run_after('setup')
    def set_executable_opts(self):
        self.executable_opts = ['-O',
                                '-i', self.input_file,
                                '-o', self.output_file]

    @run_after('setup')
    def set_keep_files(self):
        self.keep_files = [self.output_file]

    @sanity_function
    def set_sanity_patterns(self):
        energy = sn.extractsingle(r' Etot\s+=\s+(?P<energy>\S+)',
                                  self.output_file, 'energy', float, item=-2)
        energy_diff = sn.abs(energy - self.reference_value)
        ref_ener_diff = sn.abs(self.reference_value *
                               self.reference_difference)
        return sn.all([
            sn.assert_found(r'Final Performance Info:', self.output_file),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
