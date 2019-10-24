import os

import reframe as rfm
import reframe.utility.sanity as sn


class SdeBaseTest(rfm.RegressionTest):
    '''This test checks the values reported by Intel SDE for roofline modeling:
       - https://software.intel.com/en-us/articles/
            intel-software-development-emulator
       - https://bitbucket.org/dwdoerf/stream-ai-example/src/master/
       - https://www.nersc.gov/
            users/application-performance/measuring-arithmetic-intensity
    '''
    def __init__(self):
        super().__init__()
        self.descr = 'Roofline Analysis test with Intel SDE'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'sde')
        self.build_system = 'SingleSource'
        self.sourcepath = 'stream_mpi.c'
        self.executable = 'sde'
        self.target_executable = './stream.exe'
        self.sde = '%s.sde' % self.target_executable
        self.rpt = '%s.rpt' % self.target_executable
        self.build_system.ldflags = ['-g', '-O3', '-qopenmp', '-restrict',
                                     '-qopt-streaming-stores', 'always']
        exp = '/apps/dom/UES/jenkins/7.0.UP00/mc/easybuild/experimental'
        self.pre_run = [
            'mv %s %s' % (self.executable, self.target_executable),
            'module use %s/modules/all' % exp,
            'module load sde',
            'sde -help'
        ]
        self.sanity_patterns = sn.assert_found('Total FLOPs =', self.rpt)
        self.post_run = ['SDE/parse-sde.sh %s.* &> %s' % (self.sde, self.rpt)]
        self.maintainers = ['JG']
        self.tags = {'scs', 'external-resources'}

    @property
    @sn.sanity_function
    def arithmetic_intensity(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        bytes = sn.extractsingle(r'^--->Total Bytes = (?P<bytes>\d+)',
                                 self.rpt, 'bytes', int)
        return flops/bytes

    @property
    @sn.sanity_function
    def gflops(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        sec = sn.extractsingle(r'^Triad:\s+\d+\.\d+\s+(?P<avgtime>\d+\.\d+)',
                               self.stdout, 'avgtime', float)
        step = sn.extractsingle(r'^Each kernel will be executed (?P<step>\d+)',
                                self.stdout, 'step', int)
        return flops/(sec*step*10**9)

    def setup(self, partition, environ, **job_opts):
        self.executable_opts = self.sdeflags
        super().setup(partition, environ, **job_opts)
        if self.num_tasks != 36:
            self.job.options = ['--cpu-bind=verbose,none']
        else:
            self.job.options = ['--cpu-bind=verbose']


@rfm.parameterized_test(*[[num_ranks, arraysize]
                        for num_ranks in [2]
                        for arraysize in [100000000]])
# For parameter space study, you may want to use:
# for num_ranks in [36, 18, 12, 9, 6, 4, 3, 2, 1]
# for arraysize in [400000000, 200000000, 100000000]])
class SdeBroadwellJ1Test(SdeBaseTest):
    def __init__(self, num_ranks, arraysize):
        super().__init__()
        ompthread = 36 // num_ranks
        self.valid_systems = ['dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.build_system.cppflags = [
            '-D_SDE',
            '-DSTREAM_ARRAY_SIZE=%s' % arraysize,
            '-DNTIMES=50'
        ]
        self.exclusive = True
        self.num_tasks = num_ranks
        self.num_tasks_per_node = num_ranks
        self.num_cpus_per_task = ompthread
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.name = 'sde_n.{:010d}_MPI.{:03d}_OpenMP.{:03d}_j.{:01d}'.format(
            arraysize, num_ranks, ompthread, self.num_tasks_per_core)
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        self.sdeflags = ['%s -d -iform 1 -omix %s -i -top_blocks 500 '
                         '-global_region -start_ssc_mark 111:repeat '
                         '-stop_ssc_mark 222:repeat -- %s' %
                         ('-bdw', self.sde, self.target_executable)]
        # References for Intel Broadwell CPU (E5-2695 v4):
        ai = 0.0825
        gflops = 9.773
        self.sanity_patterns = sn.all([
            sn.assert_reference(self.gflops, gflops, -0.1, 0.3),
            sn.assert_reference(self.arithmetic_intensity, ai, -0.1, 0.3),
        ])


@rfm.parameterized_test(*[[num_ranks, arraysize]
                        for num_ranks in [2]
                        for arraysize in [100000000]])
# For parameter space study, you may want to use:
# for num_ranks in [72, 36, 24, 18, 12, 9, 8, 6, 4, 3, 2,
#                 1]
# for arraysize in [400000000, 200000000, 100000000]])
class SdeBroadwellJ2Test(SdeBaseTest):
    def __init__(self, num_ranks, arraysize):
        super().__init__()
        ompthread = 72 // num_ranks
        self.valid_systems = ['dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.build_system.cppflags = [
            '-D_SDE',
            '-DSTREAM_ARRAY_SIZE=%s' % arraysize,
            '-DNTIMES=50'
        ]
        self.exclusive = True
        self.num_tasks = num_ranks
        self.num_tasks_per_node = num_ranks
        self.num_cpus_per_task = ompthread
        self.num_tasks_per_core = 2
        self.use_multithreading = True
        self.name = 'sde_n.{:010d}_MPI.{:03d}_OpenMP.{:03d}_j.{:01d}'.format(
            arraysize, num_ranks, ompthread, self.num_tasks_per_core)
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        self.sdeflags = ['%s -d -iform 1 -omix %s -i -top_blocks 500 '
                         '-global_region -start_ssc_mark 111:repeat '
                         '-stop_ssc_mark 222:repeat -- %s' %
                         ('-bdw', self.sde, self.target_executable)]
        # References for Intel Broadwell CPU (E5-2695 v4):
        ai = 0.0822
        gflops = 9.602
        self.sanity_patterns = sn.all([
            sn.assert_reference(self.gflops, gflops, -0.1, 0.3),
            sn.assert_reference(self.arithmetic_intensity, ai, -0.1, 0.3),
        ])
