# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility as util
import reframe.utility.sanity as sn


@rfm.simple_test
class gromacs_check(rfm.RunOnlyRegressionTest):
    '''Base class for the Gromacs Test.

    GROMACS is a versatile package to perform molecular dynamics,
    i.e. simulate the Newtonian equations of motion for systems
    with hundreds to millions of particles.

    It is primarily designed for biochemical molecules like proteins,
    lipids and nucleic acids that have a lot of complicated bonded
    interactions, but since GROMACS is extremely fast at calculating
    the nonbonded interactions (that usually dominate simulations)
    many groups are also using it for research on non-biological
    systems, e.g. polymers (see gromacs.org).

    '''

    #: Parameter pack encoding the benchmark information.
    #:
    #: The first element of the tuple refers to the benchmark name,
    #: the second is the energy reference and the third is the
    #: tolerance threshold.
    #:
    #: :type: `Tuple[str, float, float]`
    #: :values:
    benchmark_info = parameter([
        ('HECBioSim/Crambin', 0.0, 0.0),
        ('HECBioSim/Glutamine-Binding-Protein', 0.0, 0.0),
        ('HECBioSim/hEGFRDimer', 0.0, 0.0),
        ('HECBioSim/hEGFRDimerPair', 0.0, 0.0),
        ('HECBioSim/hEGFRtetramerPair', 0.0, 0.0)
    ])

    #: Parameter encoding the implementation of the non-bonded calculations
    #:
    #: :type: :class:`str`
    #: :values: ``['cpu', 'gpu']``
    nb_impl = parameter(['cpu', 'gpu'])

    executable = 'gmx_mpi'
    tags = {'sciapp', 'chemistry'}
    keep_files = ['md.log']

    @run_after('init')
    def prepare_test(self):
        self.__bench, self.__nrg_ref, self.__nrg_tol = self.benchmark_info
        self.descr = f'GROMACS {self.__bench} benchmark (NB: {self.nb_impl})'
        self.prerun_cmds = [
            f'curl -LJO https://github.com/victorusu/GROMACS_Benchmark_Suite/raw/main/{self.__bench}/benchmark.tpr'  # noqa: E501
        ]
        self.executable_opts = ['mdrun', '-nb', self.nb_impl,
                                '-s benchmark.tpr']

    @property
    def bench_name(self):
        '''The benchmark name.

        :type: :class:`str`
        '''

        return self.__bench

    @property
    def energy_ref(self):
        '''The energy reference value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_ref

    @property
    def energy_tol(self):
        '''The energy tolerance value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_tol

    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                'md.log', 'perf', float)

    @deferrable
    def energy_hecbiosim_crambin(self):
        return sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                r'\s+Conserved En\.\s+Temperature\n'
                                r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                'md.log', 'energy', float, item=-1)

    @deferrable
    def energy_hecbiosim_glutamine_binding_protein(self):
        return sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                r'\s+Conserved En\.\s+Temperature\n'
                                r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                'md.log', 'energy', float, item=-1)

    @deferrable
    def energy_hecbiosim_hegfrdimer(self):
        return sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                r'\s+Conserved En\.\s+Temperature\n'
                                r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                'md.log', 'energy', float, item=-1)

    @deferrable
    def energy_hecbiosim_hegfrdimerpair(self):
        return sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                r'\s+Conserved En\.\s+Temperature\n'
                                r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                'md.log', 'energy', float, item=-1)

    @deferrable
    def energy_hecbiosim_hegfrtetramerpair(self):
        return sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                r'\s+Conserved En\.\s+Temperature\n'
                                r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                'md.log', 'energy', float, item=-1)

    @sanity_function
    def assert_energy_readout(self):
        '''Assert the obtained energy meets the specified tolerances.'''

        energy_fn_name = f'energy_{util.toalphanum(self.__bench).lower()}'
        energy_fn = getattr(self, energy_fn_name, None)
        sn.assert_true(
            energy_fn is not None,
            msg=(f"cannot extract energy for benchmark {self.__bench!r}: "
                 f"please define a member function '{energy_fn_name}()'")
        ).evaluate()
        energy = energy_fn()
        energy_diff = sn.abs(energy - self.energy_ref)
        return sn.all([
            sn.assert_found('Finished mdrun', 'md.log'),
            sn.assert_lt(energy_diff, self.energy_tol)
        ])
