# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class QuantumESPRESSO(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the Quantum Espresso Test.

    Quantum ESPRESSO is an integrated suite of Open-Source computer
    codes for electronic-structure calculations and materials
    modeling at the nanoscale. It is based on density-functional
    theory, plane waves, and pseudopotentials.
    (see quantum-espresso.org)

    The presented abstract run-only class checks the perfomance of
    Quantum ESPRESSO. To do this, it is necessary to define in tests
    the name of the running script (input file), as well as set the
    reference values of energy and possible deviations from this value.
    This data is used to check if the task is being executed
    correctly, that is, the final energy is correct (approximately
    the reference). The default assumption is that Quantum ESPRESSO
    is already installed on the device under test.
    '''

    #: Name of executed script
    #:
    #: :default: :class:`required`
    input_file = variable(str)

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

    executable = 'pw.x'
    input_file = 'ausurf.in'

    @run_after('setup')
    def set_executable_opts(self):
        '''Set the executable options for the Quantum ESPRESSO. Determine the
        using of input file.
        '''
        self.executable_opts = ['-in', self.input_file]

    @performance_function('s', perf_key='time')
    def set_perf_patterns(self):
        return sn.extractsingle(r'electrons.+\s(?P<wtime>\S+)s WALL',
                                self.stdout, 'wtime', float)

    @run_before('performance')
    def set_the_performance_dict(self):
        self.perf_variables = {self.mode:
                               sn.make_performance_function(
                                   sn.extractsingle(
                                       r'electrons.+\s(?P<wtime>\S+)s WALL',
                                       self.stdout, 'wtime', float), 's')}

    @sanity_function
    def set_sanity_patterns(self):
        '''Assert the obtained energy meets the specified tolerances.'''

        self.sanity_patterns = sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
        ])

        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)

        energy_diff = sn.abs(energy-self.energy_value)
        ref_ener_diff = sn.abs(self.energy_tolerance)
        return sn.all([
            self.sanity_patterns,
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
