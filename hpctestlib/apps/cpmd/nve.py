# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class Cpmd_NVE(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the CPMD NVE Test.

    The CPMD code is a plane wave/pseudopotential implementation
    of Density Functional Theory, particularly designed
    for ab-initio molecular dynamics. Its first version was
    developed by Jürg Hutter at IBM Zurich Research
    Laboratory starting from the original Car-Parrinello
    codes.

    The presented abstract run-only class checks the perfomance of CPMD.
    To do this, it is necessary to define in tests the name
    of executable file (executable), the running script
    (input file), the output file, as well as set the reference
    values of energy and possible deviations from this value.
    This data is used to check if the task is being executed
    correctly, that is, the final energy is correct
    (approximately the reference). The default assumption is that
    CPMD is already installed on the device under test.
    '''

    #: CPMD input file. This file is set by the post-init hook
    #: :func:`set_executable_opts`.
    #:
    #: :default: :class:`required`
    input_file = variable(str)

    #: CPMD output file. This file is set by the post-init hook
    #: :func:`set_executable_opts`.
    #:
    #: :default: : 'stdout.txt'
    output_file = variable(str, value='stdout.txt')

    #: Reference value of energy, that is used for the comparison
    #: with the execution ouput on the sanity step. The absolute
    #: difference between final energy value and reference value
    #: should be smaller than energy_tolerance
    #:
    #: :type: float
    #: :default: :class:`required`
    energy_value = variable(float)

    #: Maximum deviation from the reference value of energy,
    #: that is acceptable.
    #:
    #: :type: float
    #: :default: :class:`required`
    energy_tolerance = variable(float)

    #: :default: :class:`required`
    num_tasks_per_node = required

    #: :default: :class:`required`
    executable = required

    executable = 'cpmd.x'
    input_file = 'ana_c4h6.in'
    energy_value = 25.81
    energy_tolerance = 0.26
    readonly_files = ['ana_c4h6.in', 'C_MT_BLYP', 'H_MT_BLYP']

    @performance_function('s', perf_key='time')
    def set_perf_patterns(self):
        return sn.extractsingle(r'^ cpmd(\s+[\d\.]+){3}\s+(?P<perf>\S+)',
                                'stdout.txt', 'perf', float)

    @run_before('performance')
    def set_the_performance_dict(self):
        self.perf_variables = {self.mode:
                               sn.make_performance_function(
                                   sn.extractsingle(
                                       r'^ cpmd(\s+[\d\.]+){3}\s+(?P<perf>'
                                       r'\S+)', 'stdout.txt', 'perf',
                                       float), 's')}

    @run_after('setup')
    def set_executable_opts(self):
        '''Set the executable options for the CPMD. Determine the
        using of input and ouput files.
        '''
        self.executable_opts = [f'{self.input_file} > {self.output_file}']

    @sanity_function
    def set_sanity_patterns(self):
        '''Assert the obtained energy meets the specified tolerances.'''

        energy = sn.extractsingle(
            r'CLASSICAL ENERGY\s+-(?P<result>\S+)',
            'stdout.txt', 'result', float)
        energy_diff = sn.abs(energy - self.energy_value)
        ref_ener_diff = sn.abs(self.energy_tolerance)
        return sn.assert_lt(energy_diff, ref_ener_diff)
