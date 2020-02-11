import os

import reframe as rfm
import reframe.utility.sanity as sn


class ErtTestBase(rfm.RegressionTest):
    '''
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    '''

    def __init__(self):
        super().__init__()
        self.descr = 'Empirical Roofline Toolkit'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'ert')
        self.build_system = 'SingleSource'
        self.sourcepath = 'kernel1.c driver1.c'
        self.executable = 'ert.exe'
        self.build_system.ldflags = ['-O3', '-fopenmp']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'ert')
        self.rpt = '%s.rpt' % self.executable
        self.maintainers = ['JG']
        self.tags = {'scs', 'external-resources'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        if self.num_tasks != 36:
            self.job.launcher.options = ['--cpu-bind=verbose,none']


@rfm.parameterized_test(
    *[[num_ranks, flop]
      for num_ranks in [36, 18, 12, 9, 6, 4, 3, 2, 1]
      for flop in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]])
class ErtBroadwellTest(ErtTestBase):
    def __init__(self, num_ranks, flop):
        super().__init__()
        ompthread = 36 // num_ranks
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.build_system.cppflags = [
            '-DERT_FLOP=%s' % flop,
            '-DERT_ALIGN=32',
            '-DERT_MEMORY_MAX=1073741824',
            '-DERT_MPI=True',
            '-DERT_OPENMP=True',
            '-DERT_TRIALS_MIN=1',
            '-DERT_WORKING_SET_MIN=1',
        ]
        self.name = 'ert_FLOPS.{:04d}_MPI.{:03d}_OpenMP.{:03d}'.format(
            flop, num_ranks, ompthread)
        self.exclusive = True
        self.num_tasks = num_ranks
        self.num_tasks_per_node = num_ranks
        self.num_cpus_per_task = ompthread
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }

        # take the "slowest" job, make it sleep after it has ended and hope the
        # other jobs have ended too
        # TODO: find a better way to wait for the other jobs to end
        num_ranks_min = 1
        flop_min = 1024
        self.roofline_rpt = 'rpt'
        if num_ranks == num_ranks_min and flop == flop_min:
            self.post_run = [
                'cat *_job.out | python2 preprocess.py > pre',
                'python2 maximum.py < pre > max',
                'python2 summary.py < max > sum',
                # give enough time for all the dependent jobs to collect data:
                'sleep 60',
                'cat ../ert_FLOPS*/sum | python2 roofline.py > rpt',
            ]

        else:
            self.post_run = [
                'cat *_job.out | python2 preprocess.py > pre',
                'python2 maximum.py < pre > max',
                'python2 summary.py < max > sum',
            ]

        # --- Sanity check:
        regex_datatype = (r'^\s+(?P<type>\w+) \* __restrict__ buf = '
                          r'\(\w+ \*\)malloc\(PSIZE\);')
        datatype = sn.extractsingle(regex_datatype, 'driver1.c', 'type')
        self.sanity_patterns = sn.all([
            sn.assert_found('GFLOPs', 'sum'),
            sn.assert_eq(datatype, 'double'),
        ])

        # --- Performance check:
        if num_ranks == num_ranks_min and flop == flop_min:
            # Reference roofline boundaries for Intel BroadwellCPU (E5-2695v4):
            ref_GFLOPs = 945.0
            ref_L1bw = 1788.0
            ref_L2bw = 855.0
            ref_L3bw = 547.0
            ref_DRAMbw = 70.5

            # Typical performance report looks like:
            # --------------------------------------
            # ert_FLOPS.1024_MPI.001_OpenMP.036/rpt
            #    908.43 GFLOPs EMP
            #    ******
            # META_DATA
            # OPENMP_THREADS 1
            # FLOPS          8
            # MPI_PROCS      36
            #
            #   5647.33 L1 EMP
            #   *******
            #   3203.86 L2 EMP
            #   *******
            #   1773.58 L3 EMP
            #   *******
            #    139.56 L4 EMP
            #    103.50 DRAM EMP
            #    ******
            # META_DATA
            # FLOPS          2
            # OPENMP_THREADS 1
            # MPI_PROCS      36
            regex_gflops = r'(?P<GFLOPs>\d+.\d+)\sGFLOPs EMP'
            regex_L1bw = r'(?P<L1bw>\d+.\d+)\sL1 EMP'
            regex_L2bw = r'(?P<L2bw>\d+.\d+)\sL2 EMP'
            regex_L3bw = r'(?P<L3bw>\d+.\d+)\sL3 EMP'
            regex_DRAMbw = r'(?P<DRAMbw>\d+.\d+) DRAM EMP'

            gflops = sn.extractsingle(regex_gflops, self.roofline_rpt,
                                      'GFLOPs', float)
            L1bw = sn.extractsingle(regex_L1bw, self.roofline_rpt,
                                    'L1bw', float)
            L2bw = sn.extractsingle(regex_L2bw, self.roofline_rpt,
                                    'L2bw', float)
            L3bw = sn.extractsingle(regex_L3bw, self.roofline_rpt,
                                    'L3bw', float)
            DRAMbw = sn.extractsingle(regex_DRAMbw, self.roofline_rpt,
                                      'DRAMbw', float)

            # --performance-report:
            self.perf_patterns = {
                'gflops': gflops,
                'L1bw': L1bw,
                'L2bw': L2bw,
                'L3bw': L3bw,
                'DRAMbw': DRAMbw,
            }

            self.reference = {
                '*': {
                    'gflops': (ref_GFLOPs, -0.1, 0.5, 'GF/s'),
                    'L1bw': (ref_L1bw, -0.1, 0.3, 'GB/s'),
                    'L2bw': (ref_L2bw, -0.1, 0.3, 'GB/s'),
                    'L3bw': (ref_L3bw, -0.1, 0.3, 'GB/s'),
                    'DRAMbw': (ref_DRAMbw, -0.1, 0.3, 'GB/s'),
                }
            }
