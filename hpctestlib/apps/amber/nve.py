# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import reframe as rfm
import reframe.utility.sanity as sn


class Amber_NVE(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the Amber NVE Test.

    Amber is a suite of biomolecular simulation programs. It
    began in the late 1970's, and is maintained by an active
    development community (see ambermd.org).

    The presented abstract run-only class checks the amber perfomance.
    To do this, it is necessary to define in tests the name
    of the running script (input file), the output file,
    as well as set the reference values of energy and possible
    deviations from this value. This data is used to check if
    the task is being executed correctly, that is, the final energy
    is correct (approximately the reference). The default assumption
    is that Amber is already installed on the device under test.
    '''

    #: Amber output file.
    #:
    #: :default: : 'amber.out'
    output_file = variable(str, value='amber.out')

    #: Amber input file. This file is set by the post-init hook
    #: :func:`unpack_platform_parameter`.
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

    #: Parameter pack containing the platform ID, input file and
    #: executable.
    platform = parameter(['cpu', 'gpu'])

    #: NVE simulation parameter pack with the benchmark name,
    #: energy reference and energy tolerance for each case.
    benchmark = parameter([
        'Cellulose_production_NVE',
        'FactorIX_production_NVE',
        'JAC_production_NVE_4fs',
        'JAC_production_NVE'
    ])
    tags = {'sciapp', 'chemistry'}

    @run_after('init')
    def prepare_test(self):
        params = {
            'cpu': ('mdin.CPU', 'pmemd.MPI'),
            'gpu': ('mdin.GPU', 'pmemd.cuda.MPI')
        }
        with contextlib.suppress(KeyError):
            self.input_file, self.executable = params[self.platform]

        energies = {
            'Cellulose_production_NVE': (-443246.0, 5.0E-05),
            'FactorIX_production_NVE': (-234188.0, 1.0E-04),
            'JAC_production_NVE_4fs': (-44810.0, 1.0E-03),
            'JAC_production_NVE': (-58138.0, 5.0E-04)
        }
        with contextlib.suppress(KeyError):
            self.energy_value, self.energy_tolerance = energies[self.benchmark]

        self.prerun_cmds = [
            # cannot use wget because it is not installed on eiger
            f'curl -LJO https://github.com/victorusu/amber_benchmark_suite'
            f'/raw/main/amber_16_benchmark_suite/PME/{self.benchmark}.tar.bz2',
            f'tar xf {self.benchmark}.tar.bz2'
        ]
        self.executable_opts = ['-O',
                                '-i', self.input_file,
                                '-o', self.output_file]
        self.keep_files = [self.output_file]

    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'ns/day =\s+(?P<perf>\S+)',
                                self.output_file, 'perf', float, item=1)

    @sanity_function
    def assert_energy_readout(self):
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
