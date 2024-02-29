"""ReFrame benchmark for QuantumESPRESSO"""
import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import (performance_function, run_after, run_before,
                                   sanity_function)
from reframe.core.parameters import TestParam as parameter
from reframe.core.variables import TestVar as variable

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

    The benchmarks consist on a set of different inputs files ..."""

    # Tests the performance of the FFTW algorithm, higher ecut -> more FFTs
    ecut = parameter([50,150], loggable=True)
    # Tests the performance of the diagonalization algorithm, higher nbnd -> bigger matrices
    nbnd = parameter([10,200], loggable=True)

    executable = 'pw.x'
    tags = {'sciapp', 'chemistry'}

    descr = 'QuantumESPRESSO pw.x benchmark'

    input_name: str = variable(str, value='Si.scf.in')
    pp_name: str = variable(str, value='Si.pbe-n-kjpaw_psl.1.0.0.UPF')

    @run_after('init')
    def prepare_test(self):
        """Hook to the set the downloading of the pseudo-potentials"""
        self.prerun_cmds = [
            f'wget -q http://pseudopotentials.quantum-espresso.org/upf_files/{self.pp_name}'
        ]
        self.executable_opts += [f'-in {self.input_name}']

    @run_after('setup')
    def write_input(self):
        """Write the input file for the calculation"""
        inp_file = os.path.join(self.stagedir, self.input_name)
        with open(inp_file, 'w', encoding='utf-8') as file:
            file.write(INPUT_TEMPLATE.format(
                ecut=self.ecut,
                nbnd=self.nbnd,
                pseudo=self.pp_name,
                ))


    @performance_function('s')
    def extract_report_time(self, name: str = None, kind: str = None) -> float:
        """Extract timings from pw.x stdout

        Args:
            name (str, optional): Name of the timing to extract. Defaults to None.
            kind (str, optional): Kind ('cpu' or 'wall) of timing to extract. Defaults to None.

        Raises:
            ValueError: If the kind is not 'cpu' or 'wall'

        Returns:
            float: The timing in seconds
        """
        kind = kind.lower()
        if kind == 'cpu':
            tag = 1
        elif kind == 'wall':
            tag = 2
        else:
            raise ValueError(f'unknown kind: {kind}')

        return sn.extractsingle(
            fr'{name}\s+:\s+([\d\.]+)s\s+CPU\s+([\d\.]+)s\s+WALL', self.stdout, tag, float
            )

    @run_before('performance')
    def set_perf_variables(self):
        """Build a dictionary of performance variables"""

        for name in ['PWSCF', 'electrons', 'c_bands', 'sum_bands', 'cegterg', 'calbec', 'fft', 'ffts', 'fftw']:
            for kind in ['cpu', 'wall']:
                self.perf_variables[f'{name}_{kind}'] = self.extract_report_time(name, kind)

    @sanity_function
    def assert_job_finished(self):
        """Check if the job finished successfully"""
        return sn.assert_found(r'JOB DONE', self.stdout)
