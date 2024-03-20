"""ReFrame benchmark for QuantumESPRESSO"""
import os
from typing import TypeVar

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import (performance_function, run_after, run_before,
                                   sanity_function)
from reframe.core.exceptions import SanityError
from reframe.core.parameters import TestParam as parameter
from reframe.core.variables import TestVar as variable

R = TypeVar('R')

INPUT_TEMPLATE = """&CONTROL
  calculation  = "scf",
  prefix       = "Si",
  pseudo_dir   = ".",
  outdir       = "./out",
  restart_mode = "from_scratch"
  verbosity    = 'high'
/
&SYSTEM
  ibrav     = 2,
  celldm(1) = 10.2,
  nat       = 2,
  ntyp      = 1,
  nbnd      = {nbnd}
  ecutwfc   = {ecut}
/
&ELECTRONS
  conv_thr    = 1.D-8,
  mixing_beta = 0.7D0,
/
ATOMIC_SPECIES
 Si  28.086  {pseudo}
ATOMIC_POSITIONS
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS {{automatic}}
 15 15 15   0 0 0

"""


@rfm.simple_test
class QEspressoPWCheck(rfm.RunOnlyRegressionTest):
    """QuantumESPRESSO benchmark test.

    `QuantumESPRESSO <https://www.quantum-espresso.org/>`__ is an integrated
    suite of Open-Source computer codes for electronic-structure calculations
    and materials modeling at the nanoscale.

    The benchmarks consist of one input file templated inside the code and
    a pseudo-potential file that is downloaded from the official repository.

    This tests aims at measuring the scalability of the pw.x executable, in
    particular the FFT and diagonalization algorithms, by running a simple
    silicon calculation with high `ecut` (increases size of FFTs) and `nbnd`
    (increases size of matrices to diagonalize) values."""

    #: Parametert to tests the performance of the FFTW algorithm,
    #: higher `ecut` implicates more FFTs
    #:
    #: :type: :class:`int`
    #: :values: ``[50, 150]``
    ecut = parameter([50, 150], loggable=True)

    #: Parameter to Tests the performance of the diagonalization algorithm,
    #: higher `nbnd` implicates bigger matrices
    #:
    #: :type: :class:`int`
    #: :values: ``[10, 200]``
    nbnd = parameter([10, 200], loggable=True)

    executable = 'pw.x'
    tags = {'sciapp', 'chemistry'}

    descr = 'QuantumESPRESSO pw.x benchmark'

    #: The name of the input file used.
    #:
    #: :type: :class:`str`
    #: :default: ``'Si.scf.in'``
    input_name: str = variable(str, value='Si.scf.in')

    #: The pseudo-potential file to be used check
    #: https://www.quantum-espresso.org/pseudopotentials/ for more info
    #:
    #: :type: :class:`str`
    #: :default: ``'Si.pbe-n-kjpaw_psl.1.0.0.UPF'``
    pp_name: str = variable(str, value='Si.pbe-n-kjpaw_psl.1.0.0.UPF')

    @run_after('init')
    def prepare_test(self):
        """Hook to the set the downloading of the pseudo-potentials"""
        self.prerun_cmds = [(
            'wget -q http://pseudopotentials.quantum-espresso.org/upf_files/'
            f'{self.pp_name}'
        )]
        self.executable_opts += [f'-in {self.input_name}']

    @run_after('setup')
    def write_input(self):
        """Write the input file for the calculation"""
        inp_file = os.path.join(self.stagedir, self.input_name)
        with open(inp_file, 'w', encoding='utf-8') as file:
            file.write(
                INPUT_TEMPLATE.format(
                    ecut=self.ecut,
                    nbnd=self.nbnd,
                    pseudo=self.pp_name,
                ))

    @staticmethod
    @sn.deferrable
    def extractsingle_or_val(*args, on_except_value: str = '0s') -> str:
        """Wrap extractsingle_or_val to return a default value if the regex is
        not found.

        Returns:
            str: The value of the regular expression
        """
        try:
            res = sn.extractsingle(*args).evaluate()
        except SanityError:
            res = on_except_value

        return res

    @staticmethod
    @sn.deferrable
    def convert_timings(timing: str) -> float:
        """Convert timings to seconds"""

        if timing is None:
            return 0

        days, timing = (['0', '0'] + timing.split('d'))[-2:]
        hours, timing = (['0', '0'] + timing.split('h'))[-2:]
        minutes, timing = (['0', '0'] + timing.split('m'))[-2:]
        seconds = timing.split('s')[0]

        return (
            float(days) * 86400 +
            float(hours) * 3600 +
            float(minutes) * 60 +
            float(seconds)
        )

    @performance_function('s')
    def extract_report_time(self, name: str = None, kind: str = None) -> float:
        """Extract timings from pw.x stdout

        Args:
            name (str, optional): Name of the timing to extract.
                                  Defaults to None.
            kind (str, optional): Kind ('cpu' or 'wall) of timing to extract.
                                  Defaults to None.

        Raises:
            ValueError: If the kind is not 'cpu' or 'wall'

        Returns:
            float: The timing in seconds
        """
        if kind is None:
            return 0
        kind = kind.lower()
        if kind == 'cpu':
            tag = 1
        elif kind == 'wall':
            tag = 2
        else:
            raise ValueError(f'unknown kind: {kind}')

        # Possible formats
        #       PWSCF        :   4d 6h19m CPU  10d14h38m WALL
        # --> (Should also catch spaces)
        return self.convert_timings(
            self.extractsingle_or_val(
                fr'{name}\s+:\s+(.+)\s+CPU\s+(.+)\s+WALL',
                self.stdout, tag, str
            ))

    @run_before('performance')
    def set_perf_variables(self):
        """Build a dictionary of performance variables"""

        timings = [
            'PWSCF', 'electrons', 'c_bands', 'cegterg', 'calbec',
            'fft', 'ffts', 'fftw'
        ]
        for name in timings:
            for kind in ['cpu', 'wall']:
                res = self.extract_report_time(name, kind)
                self.perf_variables[f'{name}_{kind}'] = res

    @sanity_function
    def assert_job_finished(self):
        """Check if the job finished successfully"""
        return sn.assert_found(r'JOB DONE', self.stdout)
