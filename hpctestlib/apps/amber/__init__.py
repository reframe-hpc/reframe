#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["Amber"]


class Amber(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the Amber Test.

    Amber is a suite of biomolecular simulation programs. It
    began in the late 1970's, and is maintained by an active
    development community (see ambermd.org).

    The presented abstract run-only class checks the work of amber.
    To do this, it is necessary to define in tests the name
    of the running script (input file), the output file,
    as well as set the reference values of energy and possible
    deviations from this value. This data is used to check if
    the task is being executed correctly, that is, the final energy
    is correct (approximately the reference). The default assumption
    is that Amber is already installed on the device under test.
    '''

    #: Amber input file.
    #:
    #: :default: : 'amber.out'
    output_file = variable(str, value='amber.out')

    #: Reference value of energy, that is used for the comparison
    #: with the execution ouput on the sanity step. Final value of
    #: energy should be approximately the same
    #:
    #: :default: :class:`required`
    energy_value = variable(float)

    #: Maximum deviation from the reference value of energy,
    #: that is acceptable.
    #:
    #: :default: :class:`required`
    energy_tolerance = variable(float)

    num_tasks_per_node = required
    executable = required

    @run_after('setup')
    def set_keep_files(self):
        self.keep_files = [self.output_file]

    @run_before('run')
    def set_executable_opts(self):
        '''Set the executable options for the Amber. Determine the
        using of input and ouput files.
        '''
        self.executable_opts = ['-O',
                                '-i', self.input_file,
                                '-o', self.output_file]

    @sanity_function
    def set_sanity_patterns(self):
        '''Assert the obtained energy meets the specified tolerances.'''

        energy = sn.extractsingle(r' Etot\s+=\s+(?P<energy>\S+)',
                                  self.output_file, 'energy', float, item=-2)
        energy_diff = sn.abs(energy - self.energy_value)
        ref_ener_diff = sn.abs(self.energy_value *
                               self.energy_tolerance)
        return sn.all([
            sn.assert_found(r'Final Performance Info:', self.output_file),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
