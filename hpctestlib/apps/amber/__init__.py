#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ

__all__ = ["Amber_NVE"]


class Amber_NVE(rfm.RunOnlyRegressionTest, pin_prefix=True):
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

    #: Amber output file.
    #:
    #: :default: : 'amber.out'
    output_file = variable(str, value='amber.out')

    #: Amber input file.
    #:
    #: :default: : 'amber.out'
    input_file = variable(str)

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

    executable_files = parameter([
        ('cpu', 'mdin.CPU', 'pmemd.MPI'),
        ('gpu', 'mdin.GPU', 'pmemd.cuda.MPI')
    ])

    # NVE simulations
    variant = parameter([
        ('Cellulose_production_NVE', -443246.0, 5.0E-05),
        ('FactorIX_production_NVE', -234188.0, 1.0E-04),
        ('JAC_production_NVE_4fs', -44810.0, 1.0E-03),
        ('JAC_production_NVE', -58138.0, 5.0E-04)
    ])

    num_tasks_per_node = required
    executable = required

    @run_after('init')
    def unpack_platform_parameter(self):
        self.platform, self.input_file, self.executable = self.executable_files

    @run_after('init')
    def unpack_variant_parameter(self):
        (self.benchmark, self.energy_value,
            self.energy_tolerance) = self.variant

    @run_after('setup')
    def set_keep_files(self):
        self.keep_files = [self.output_file]

    @run_after('setup')
    def set_perf_patterns(self):
        self.perf_patterns = {
            self.benchmark: sn.extractsingle(r'ns/day =\s+(?P<perf>\S+)',
                                             self.output_file, 'perf',
                                             float, item=1)
        }

    @run_before('performance')
    def set_generic_perf_references(self):
        self.reference.update({'*': {
            self.benchmark: (0, None, None, 'ns/day')
        }})

    @run_before('run')
    def download_files(self):
        self.prerun_cmds = [
            # cannot use wget because it is not installed on eiger
            f'curl -LJO https://github.com/victorusu/amber_benchmark_suite'
            f'/raw/main/amber_16_benchmark_suite/PME/{self.benchmark}.tar.bz2',
            f'tar xf {self.benchmark}.tar.bz2'
        ]

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
