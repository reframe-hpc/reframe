# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class amber_nve_check(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Amber NVE test.

    `Amber <https://ambermd.org/>`__ is a suite of biomolecular simulation
    programs. It began in the late 1970's, and is maintained by an active
    development community.

    This test is parametrized over the benchmark type (see
    :attr:`benchmark_info`) and the variant of the code (see :attr:`variant`).
    Each test instance executes the benchmark, validates numerically its output
    and extracts and reports a performance metric.

    '''

    #: The output file to pass to the Amber executable.
    #:
    #: :type: :class:`str`
    #: :required: No
    #: :default: ``'amber.out'``
    output_file = variable(str, value='amber.out')

    #: The input file to use.
    #:
    #: This is set to ``mdin.CPU`` or ``mdin.GPU`` depending on the test
    #: variant during initialization.
    #:
    #: :type: :class:`str`
    #: :required: Yes
    input_file = variable(str)

    #: The name of the benchmark that this test encodes.
    #:
    #: This is set from the corresponding value in the :attr:`benchmark_info`
    #: parameter pack during initialization.
    #:
    #: :type: :class:`str`
    #: :required: Yes
    benchmark = variable(str)

    #: Energy value reference.
    #:
    #: This is set from the corresponding value in the :attr:`benchmark_info`
    #: parameter pack during initialization.
    #:
    #: :type: `float`
    #: :required: Yes
    energy_ref = variable(float)

    #: Energy value tolerance.
    #:
    #: This is set from the corresponding value in the :attr:`benchmark_info`
    #: parameter pack during initialization.
    #:
    #: :type: `float`
    #: :required: Yes
    energy_tol = variable(float)

    #: Parameter pack encoding the benchmark information.
    #:
    #: The first element of the tuple refers to the benchmark name,
    #: the second is the energy reference and the third is the
    #: tolerance threshold.
    #:
    #: :type: `Tuple[str, float, float]`
    #: :values:
    #:     .. code-block:: python
    #:
    #:        [
    #:            ('Cellulose_production_NVE', -443246.0, 5.0E-05),
    #:            ('FactorIX_production_NVE', -234188.0, 1.0E-04),
    #:            ('JAC_production_NVE_4fs', -44810.0, 1.0E-03),
    #:            ('JAC_production_NVE', -58138.0, 5.0E-04)
    #:        ]
    benchmark_info = parameter([
        ('Cellulose_production_NVE', -443246.0, 5.0E-05),
        ('FactorIX_production_NVE', -234188.0, 1.0E-04),
        ('JAC_production_NVE_4fs', -44810.0, 1.0E-03),
        ('JAC_production_NVE', -58138.0, 5.0E-04)
    ])

    # Parameter encoding the variant of the test.
    #
    # :type:`str`
    # :values: ``['mpi', 'cuda']``
    variant = parameter(['mpi', 'cuda'])

    # Test tags
    #
    # :required: No
    # :default: ``{'sciapp', 'chemistry'}``
    tags = {'sciapp', 'chemistry'}

    #: See :attr:`~reframe.core.pipeline.RegressionTest.num_tasks`.
    #:
    #: The ``mpi`` variant of the test requires ``num_tasks > 1``.
    #:
    #: :required: Yes
    num_tasks = required

    @run_after('init')
    def prepare_test(self):
        self.benchmark, self.energy_ref, self.energy_tol = self.benchmark_info
        self.descr = f'Amber NVE {self.benchmark} benchmark ({self.variant})'

        params = {
            'mpi':  ('mdin.CPU', 'pmemd.MPI'),
            'cuda': ('mdin.GPU', 'pmemd.cuda.MPI')
        }
        try:
            self.input_file, self.executable = params[self.variant]
        except KeyError:
            raise ValueError(
                f'test not set up for platform {self.variant!r}'
            ) from None

        self.prerun_cmds = [
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
        '''The performance of the benchmark expressed in ``ns/day``.'''
        return sn.extractsingle(r'ns/day =\s+(?P<perf>\S+)',
                                self.output_file, 'perf', float, item=1)

    @sanity_function
    def assert_energy_readout(self):
        '''Assert that the obtained energy meets the required tolerance.'''

        energy = sn.extractsingle(r' Etot\s+=\s+(?P<energy>\S+)',
                                  self.output_file, 'energy', float, item=-2)
        energy_diff = sn.abs(energy - self.energy_ref)
        ref_ener_diff = sn.abs(self.energy_ref *
                               self.energy_tol)
        return sn.all([
            sn.assert_found(r'Final Performance Info:', self.output_file),
            sn.assert_lt(energy_diff, ref_ener_diff)
        ])
