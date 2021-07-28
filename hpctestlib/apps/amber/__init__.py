#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["AmberBaseCheck"]


class AmberBaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the Amber Test. It is adapted to check the
    correctness of the execution of a given script (in terms of
    energy received).
    '''

    # Name of executed script. Required variable
    input_file = variable(str)

    # Name of file, where the result of program execution will
    # be saved. Required variable
    output_file = variable(str)

    # Required variable
    num_tasks_per_node = required

    # Reference value of energy, that is used for the comparison.
    # Required variable
    reference_value = variable(float)

    # Maximum deviation from the reference  value of energy,
    # that is acceptable. Required variable
    reference_difference = variable(float)

    @run_after('setup')
    def set_executable_opts(self):
        '''Set the executable options for the Amber. Determine the
        using of input and ouput files.
        '''
        self.executable_opts = ['-O',
                                '-i', self.input_file,
                                '-o', self.output_file]

    @run_after('setup')
    def set_keep_files(self):
        '''Set the ouput_file as keeping file'''

        self.keep_files = [self.output_file]

    @sanity_function
    def set_sanity_patterns(self):
        '''Standart sanity check for the Amber. Compare the
        reference value of energy with obtained from the executed
        program.
        '''

        energy = sn.extractsingle(r' Etot\s+=\s+(?P<energy>\S+)',
                                  self.output_file, 'energy', float, item=-2)
        energy_diff = sn.abs(energy - self.reference_value)
        ref_ener_diff = sn.abs(self.reference_value *
                               self.reference_difference)
        return sn.all([
            sn.assert_found(r'Final Performance Info:', self.output_file),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
