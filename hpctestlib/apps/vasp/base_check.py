# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


class VASP(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the VASP Test.

    The Vienna Ab initio Simulation Package (VASP) is a computer
    program for atomic scale materials modelling, e.g. electronic
    structure calculations and quantum-mechanical molecular
    dynamics, from first principles. (see vasp.at)

    The presented abstract run-only class checks the perfomance of VASP.
    To do this, it is necessary to define in tests the reference
    values of force and possible deviations from this value.
    This data is used to check if the task is being executed
    correctly, that is, the final force is correct (approximately
    the reference). The default assumption is that VASP is already
    installed on the device under test.
    '''

    #: Reference value of force, that is used for the comparison
    #: with the execution ouput on the sanity step. The absolute
    #: difference between final force value and reference value
    #: should be smaller than force_tolerance
    #:
    #: :type: float
    #: :default: :class:`required`
    force_value = variable(float)

    #: Maximum deviation from the reference value of force,
    #: that is acceptable.
    #:
    #: :type: float
    #: :default: :class:`required`
    force_tolerance = variable(float)

    #: Name of the keep files for the case of VASP is standart
    keep_files = ['OUTCAR']

    #: :default: :class:`required`
    num_tasks_per_node = required

    force_value = -.85026214E+03
    force_tolerance = 1e-5

    #: Parameter pack containing the platform name and executable.
    platform_info = parameter([
        ('cpu', 'vasp_std'),
        ('gpu', 'vasp_gpu')
    ])

    @run_after('init')
    def unpack_platform_parameter(self):
        '''Set the executable and input file.'''

        self.platform, self.executable = self.platform_info

    @performance_function('s', perf_key='time')
    def set_perf_patterns(self):
        return sn.extractsingle(r'Total CPU time used \(sec\):'
                                r'\s+(?P<time>\S+)', 'OUTCAR',
                                'time', float)

    @run_before('performance')
    def set_the_performance_dict(self):
        self.perf_variables = {self.mode:
                               sn.make_performance_function(
                                   sn.extractsingle(
                                       r'Total CPU time used \(sec\):'
                                       r'\s+(?P<time>\S+)', 'OUTCAR',
                                       'time', float), 's')}

    @sanity_function
    def set_sanity_patterns(self):
        '''Assert the obtained energy meets the specified tolerances.'''

        force = sn.extractsingle(r'1 F=\s+(?P<result>\S+)',
                                 self.stdout, 'result', float)
        force_diff = sn.abs(force - self.force_value)
        ref_force_diff = sn.abs(self.force_value *
                                self.force_tolerance)

        return sn.assert_lt(force_diff, ref_force_diff)
