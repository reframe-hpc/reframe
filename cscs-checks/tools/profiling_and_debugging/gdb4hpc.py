import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SbucheckCommandCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:login', 'dom:login']
        self.descr = 'Cray parallel debugger'
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['gdb4hpc']

        self.executable  = 'gdb4hpc -v'
        self.executable += ';cd F90; pwd'
        self.executable += ";make -j 1 CC='cc' CXX='CC' FC='ftn' NVCC='nvcc' CFLAGS='-g -O2 -fopenmp' CXXFLAGS='-g -O2 -fopenmp' FCFLAGS='-g -O2 -fopenmp'"
        self.executable += ';ls -lrt'
        self.executable += ';gdb4hpc -b ./gdb4hpc.in &> gdb4hpc.rpt'

        self.variables = {
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic'
        }

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.tags = {'production'}

        self.sanity_patterns = sn.all([
            sn.assert_reference(sn.extractsingle(
                r'^tst\{0\}:\s+(?P<result>\d+.\d+[eE]-\d+)',
                'F90/gdb4hpc.rpt',
                'result', float), 2.572e-6, -1e-1, 1.0e-1),

            sn.assert_found(r'gdb4hpc 3.0 - Cray Line Mode Parallel Debugger',
                            'F90/gdb4hpc.rpt'),

            sn.assert_found(r'Shutting down debugger and killing application',
                            'F90/gdb4hpc.rpt')
        ])


        self.maintainers = ['JG', 'MK']

