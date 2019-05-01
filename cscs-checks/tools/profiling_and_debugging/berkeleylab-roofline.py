import os

import reframe as rfm
import reframe.utility.sanity as sn


class ErtTestBase(rfm.RegressionTest):
    """
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    """
    def __init__(self):
        super().__init__()
        self.descr = 'Empirical Roofline Toolkit'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'ert')
        self.build_system = 'SingleSource'
        self.sourcepath = 'kernel1.c driver1.c'
        self.executable = 'ert.exe'
        self.build_system.ldflags = ['-O3 -fopenmp']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'ert')
        self.rpt = '%s.rpt' % self.executable
        self.maintainers = ['JG']
        self.tags = {'scs'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)


@rfm.parameterized_test(*[[mpitask, flop]
                        for mpitask in [36, 18, 12, 9, 6, 4, 3, 2, 1]
                        for flop in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                     1024]])
class ErtBroadwellTest(ErtTestBase):
    def __init__(self, mpitask, flop):
        super().__init__()
        ompthread = int(36/mpitask)
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.build_system.cppflags = [
            '-DERT_ALIGN=32',
            '-DERT_MEMORY_MAX=1073741824',
            '-DERT_MPI=True',
            '-DERT_OPENMP=True',
            '-DERT_TRIALS_MIN=1',
            '-DERT_WORKING_SET_MIN=1',
            '-DERT_FLOP=%s' % flop,
        ]
        self.name = 'ert_FLOPS.' + '{:03d}'.format(flop) + \
                    '_MPI.' + '{:03d}'.format(mpitask) + \
                    '_OpenMP.' + '{:03d}'.format(ompthread)
        self.time_limit = (0, 10, 0)
        self.exclusive = True
        self.num_tasks = mpitask
        self.num_tasks_per_node = mpitask
        self.num_cpus_per_task = int(ompthread)
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        # Reference roofline boundaries for Intel Broadwell CPU (E5-2695 v4):
        GFLOPs = 469.0
        L1bw = 1788.0
        L2bw = 855.0
        L3bw = 547.0
        DRAMbw = 70.5
        # slowest job:
        mpitaskm1 = 1
        flopm1 = 1024
        self.roofline_rpt = 'rpt'
        if mpitask == mpitaskm1 and flop == flopm1:
            self.post_run = [
                'cat *_job.out | python2 preprocess.py > pre',
                'python2 maximum.py < pre > max',
                'python2 summary.py < max > sum',
                # give enough time for all the dependent jobs to collect data
                'sleep 60',
                'cat ../ert_FLOPS*/sum |python2 roofline.py > rpt',
            ]
            self.sanity_patterns = sn.all([
                # --- check data type:
                sn.assert_eq(sn.extractsingle(
                    r'^\s+(?P<prec>\w+) \* __restrict__ buf = \(\w+ \*\)'
                    r'malloc\(PSIZE\);', 'driver1.c', 'prec'), 'double'),
                # --- check ert's roofline results:
                # check GFLOPS:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<GFLOPs>\d+.\d+)\sGFLOPs EMP', self.roofline_rpt,
                    'GFLOPs', float), GFLOPs, -0.1, 0.3),
                # check L1 bandwidth:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<L1bw>\d+.\d+)\sL1 EMP', self.roofline_rpt,
                    'L1bw', float), L1bw, -0.1, 0.3),
                # check L2 bandwidth:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<L2bw>\d+.\d+)\sL2 EMP', self.roofline_rpt,
                    'L2bw', float), L2bw, -0.1, 0.3),
                # check L3 bandwidth:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<L3bw>\d+.\d+)\sL3 EMP', self.roofline_rpt,
                    'L3bw', float), L3bw, -0.1, 0.3),
                # check DRAM bandwidth:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<DRAMbw>\d+.\d+) DRAM EMP', self.roofline_rpt,
                    'DRAMbw', float), DRAMbw, -0.1, 0.3),
            ])
        else:
            self.post_run = [
                'cat *_job.out | python2 preprocess.py > pre',
                'python2 maximum.py < pre > max',
                'python2 summary.py < max > sum',
            ]
            self.sanity_patterns = sn.assert_found('GFLOPs', 'sum')

        if not mpitask == 36:
            self.job.launcher.options = ['--cpu-bind=verbose,none']
