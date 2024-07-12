"""ReFrame benchmark for MetalWalls"""
import re
from typing import TypeVar

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import (performance_function, run_after, run_before,
                                   sanity_function)
from reframe.core.parameters import TestParam as parameter
from reframe.core.variables import TestVar as variable

R = TypeVar('R')

address_tpl = (
    'https://gitlab.com/ampere2/metalwalls/-/raw/{version}/{bench}/{file}'
    '?ref_type=tags\&inline=false'
)

extract_fields = [
    #######
    ('Ions->Atoms Coulomb potential', 'long range', 'I->A Cpot'),
    ('Ions->Atoms Coulomb potential', 'k==0', 'I->A Cpot'),
    ('Ions->Atoms Coulomb potential', 'short range', 'I->A Cpot'),

    #######
    ('Atoms->Atoms Coulomb potential', 'long range', 'A->A Cpot'),
    ('Atoms->Atoms Coulomb potential', 'k==0', 'A->A Cpot'),
    ('Atoms->Atoms Coulomb potential', 'short range', 'A->A Cpot'),
    ('Atoms->Atoms Coulomb potential', 'self', 'A->A Cpot'),

    #######
    ('Atoms->Atoms Coulomb grad Q', 'long range', 'A->A gradQ'),
    ('Atoms->Atoms Coulomb grad Q', 'k==0', 'A->A gradQ'),
    ('Atoms->Atoms Coulomb grad Q', 'short range', 'A->A gradQ'),
    ('Atoms->Atoms Coulomb grad Q', 'self', 'A->A gradQ'),

    #######
    ('Ions Coulomb forces', 'long range', 'Ions Cfrc'),
    ('Ions Coulomb forces', 'k==0', 'Ions Cfrc'),
    ('Ions Coulomb forces', 'short range', 'Ions Cfrc'),
    ('Ions Coulomb forces', 'intramolecular', 'Ions Cfrc'),

    #######
    ('Ions Coulomb potential', 'long range', 'Ions Cpot'),
    ('Ions Coulomb potential', 'k==0', 'Ions Cpot'),
    ('Ions Coulomb potential', 'short range', 'Ions Cpot'),
    ('Ions Coulomb potential', 'intramolecular', 'Ions Cpot'),
    ('Ions Coulomb potential', 'self', 'Ions Cpot'),

    #######
    (
        'Ions Coulomb electric field (due to charges)',
        'long range',
        'Ions Cfield(charges)'
    ),
    (
        'Ions Coulomb electric field (due to charges)',
        'k==0',
        'Ions Cfield(charges)'
    ),
    (
        'Ions Coulomb electric field (due to charges)',
        'short range',
        'Ions Cfield(charges)'
    ),
    (
        'Ions Coulomb electric field (due to charges)',
        'intramolecular',
        'Ions Cfield(charges)'
    ),

    #######
    (
        'Ions Coulomb electric field (due to dipoles)',
        'long range',
        'Ions Cfield(dipoles)'
    ),
    (
        'Ions Coulomb electric field (due to dipoles)',
        'k==0',
        'Ions Cfield(dipoles)'
    ),
    (
        'Ions Coulomb electric field (due to dipoles)',
        'short range',
        'Ions Cfield(dipoles)'
    ),
    (
        'Ions Coulomb electric field (due to dipoles)',
        'intramolecular',
        'Ions Cfield(dipoles)'
    ),
    (
        'Ions Coulomb electric field (due to dipoles)',
        'self',
        'Ions Cfield(dipoles)'
    ),

    #######
    ('Ions Coulomb electric field gradient', 'long range', 'Ions Cfield grad'),
    ('Ions Coulomb electric field gradient', 'short range', 'Ions Cfield grad'),
    ('Ions Coulomb electric field gradient', 'self', 'Ions Cfield grad'),

    #######
    ('Ions Coulomb gradient mu', 'long range', 'Ions C mu_grad'),
    ('Ions Coulomb gradient mu', 'k==0', 'Ions C mu_grad'),
    ('Ions Coulomb gradient mu', 'short range', 'Ions C mu_grad'),
    ('Ions Coulomb gradient mu', 'self', 'Ions C mu_grad'),

    #######
    ('Rattle', 'positions', 'Rattle'),
    ('Rattle', 'velocities', 'Rattle'),

    #######
    ('van der Waals', 'vdW  forces', 'van der Waals'),
    ('van der Waals', 'vdW  potential', 'van der Waals'),

    #######
    ('Intramolecular', 'Intramolecular  forces', 'Intramolecular'),
    ('Intramolecular', 'Intramolecular potential', 'Intramolecular'),

    #######
    ('Additional degrees', 'Electrode charge computation', 'Deg of freedom'),
    ('Additional degrees', 'Inversion of the matrix', 'Deg of freedom'),
    ('Additional degrees', 'One matrix-vector product', 'Deg of freedom'),
    ('Additional degrees', 'Melt dipoles computation', 'Deg of freedom'),
    ('Additional degrees', 'Inversion of the matrix', 'Deg of freedom'),
    ('Additional degrees', 'One matrix-vector product', 'Deg of freedom'),
    ('Additional degrees', 'AIM DOFs computation', 'Deg of freedom'),

    #######
    ('Diagnostics', 'diagnostics computations', 'Diagnostics'),
    ('Diagnostics', 'IO', 'Diagnostics'),
]

@rfm.simple_test
class MetalWallsCheck(rfm.RunOnlyRegressionTest):
    """MetalWalls benchmark test.
    
    `MetalWalls <https://gitlab.com/ampere2/metalwalls>`__ is a molecular 
    dynamics code dedicated to the modelling of electrochemical systems.
    Its main originality is the inclusion of a series of methods allowing to
    apply a constant potential within the electrode materials.

    The benchmarks consist of a set of different inputs files that vary in the
    number of atoms and the type of simulation performed.
    They can be found in the following repository, which is also versioned:
    https://gitlab.com/ampere2/metalwalls/.
    
    """

    #: The name of the output files to keep.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``'run.out'``
    keep_files = ['run.out', ]

    #: The version of the benchmark suite to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'21.06.1'``
    benchmark_version = variable(str, value='21.06.1', loggable=True)

    #: Collect and report detailed performance metrics.
    #:
    #: :type: :class:`bool`
    #: :default: ``False``
    debug_metrics = variable(bool, value=False, loggable=True)

    #: Parameter pack encoding the benchmark information.
    #:
    #: The first element of the tuple refers to the benchmark name,
    #:
    #: :type: `Tuple[str,]`
    #: :values:
    benchmark_info = parameter([
        ('hackathonGPU/benchmark',),
        ('hackathonGPU/benchmark2',),
        ('hackathonGPU/benchmark3',),
        ('hackathonGPU/benchmark4',),
        ('hackathonGPU/benchmark5',),
        ('hackathonGPU/benchmark6',),
    ], fmt=lambda x: x[0], loggable=True)

    @run_after('init')
    def prepare_test(self):
        """Hook to the set the downloading of the pseudo-potentials"""
        self.__bench, = self.benchmark_info
        self.descr = f'MetalWalls {self.__bench} benchmark'
        files_addresses = [
            address_tpl.format(
                version=self.benchmark_version,
                bench=self.__bench,
                file=_
            ) for _ in ['data.inpt', 'runtime.inpt']
        ]
        self.prerun_cmds += [
            f"curl -LJO '{_}'" for _ in files_addresses
        ]

    @performance_function('s')
    def total_elapsed_time(self):
        """Extract the total elapsed time from the output file"""
        return sn.extractsingle(
            r'Total elapsed time:\s+(?P<time>\S+)', 'run.out', 'time', float
            )

    @performance_function('s')
    def extract_time(
        self, name: str = None, parent: str = None, kind: str = None
        ) -> float:
        """Extract the time from a specific report section of the output file
        
        Args:
            name (str): The name of the report to extract
            parent (str): The parent section of the report
            kind (str): The kind of time to extract (avg or cumul)
        """
        kind = kind.lower()
        if kind == 'avg':
            tag = 1
        elif kind == 'cumul':
            tag = 2
        else:
            raise ValueError(f'Unknown kind: {kind}')
        
        # Example Fromat:
        # Ions->Atoms Coulomb potential
        # -----------------------------
        #   long range                       9.55040E-03  9.64590E-01         0.71
        #   k==0                             2.48271E-02  2.50754E+00         1.84
        #   short range                      8.72464E-02  8.81189E+00         6.46
        #
        # NEXT SECTION
        res = -1
        rgx = re.compile(rf'^ +{name}\s+(\S+) +(\S+)')
        flag = False
        with open('run.out', 'r') as file:
            for line in file:
                if flag:
                    if rgx.match(line):
                        res = float(rgx.match(line).group(tag))
                        break
                    elif not line.strip():
                        break
                else:
                    if parent in line:
                        flag = True

        return res

    @run_before('performance')
    def set_perf_variables(self):
        """Build a dictionary of performance variables"""
        self.perf_variables['total_elapsed_time'] = self.total_elapsed_time()

        if self.debug_metrics:
            for parent, name, short in extract_fields:
                name2 = name.replace(' ', '_')
                for kind in ['avg', 'cumul']:
                    app = self.extract_time(name, parent, kind)
                    self.perf_variables[f'{short}__{name2}__{kind}'] = app

    @sanity_function
    def assert_job_finished(self):
        """Check if the job finished successfully"""
        return sn.assert_found(r'Total elapsed time', 'run.out')
