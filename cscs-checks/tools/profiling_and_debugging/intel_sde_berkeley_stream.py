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
        self.pre_run = [
            'mv %s %s' % (self.executable, self.target_executable),
            'module use /apps/dom/UES/jenkins/7.0.UP00/mc/easybuild/'
            'experimental/modules/all',
            'module load sde',
            'sde -help'
        ]
        self.sanity_patterns = sn.assert_found('Total FLOPs =', self.rpt)
        self.post_run = ['./parse-sde.sh %s.* &> %s' % (self.sde, self.rpt)]
        self.maintainers = ['JG']
        self.tags = {'scs'}

    def setup(self, partition, environ, **job_opts):
        self.executable_opts = self.sdeflags
        super().setup(partition, environ, **job_opts)
        if not self.num_tasks == 36:
            self.job.options = ['--cpu-bind=verbose,none']
        else:
            self.job.options = ['--cpu-bind=verbose']


@rfm.parameterized_test(*[[mpitask, arraysize]
                        for mpitask in [2]
                        for arraysize in [100000000]])
# for mpitask in [36, 18, 12, 9, 6, 4, 3, 2, 1]
# for arraysize in [400000000, 200000000, 100000000]])
class SdeBroadwellJ1Test(SdeBaseTest):
    def __init__(self, mpitask, arraysize):
        super().__init__()
        ompthread = int(36/mpitask)
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.build_system.cppflags = [
            '-D_SDE',
            '-DSTREAM_ARRAY_SIZE=%s' % arraysize,
            '-DNTIMES=50'
        ]
        self.time_limit = (0, 10, 0)
        self.exclusive = True
        self.num_tasks = mpitask
        self.num_tasks_per_node = mpitask
        self.num_cpus_per_task = int(ompthread)
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.name = 'sde_n.' + '{:010d}'.format(arraysize) + \
                    '_MPI.' + '{:03d}'.format(mpitask) + \
                    '_OpenMP.' + '{:03d}'.format(ompthread) + \
                    '_j.%s' % self.num_tasks_per_core
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
        gflops = 9773.0
        self.sanity_patterns = sn.all([
            sn.assert_reference(self.gflops, gflops, -0.1, 0.3),
            sn.assert_reference(self.arithmetic_intensity, ai, -0.1, 0.3),
        ])

    @property
    @sn.sanity_function
    def arithmetic_intensity(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        byts = sn.extractsingle(r'^--->Total Bytes = (?P<byts>\d+)',
                                self.rpt, 'byts', int)
        return flops/byts

    @property
    @sn.sanity_function
    def gflops(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        sec = sn.extractsingle(r'^Triad:\s+\d+\.\d+\s+(?P<avgtime>\d+\.\d+)',
                               self.stdout, 'avgtime', float)
        step = sn.extractsingle(r'^Each kernel will be executed (?P<step>\d+)',
                                self.stdout, 'step', int)
        return flops/(sec*step*10**6)


@rfm.parameterized_test(*[[mpitask, arraysize]
                        for mpitask in [2]
                        for arraysize in [100000000]])
# for mpitask in [72, 36, 24, 18, 12, 9, 8, 6, 4, 3, 2,
#                 1]
# for arraysize in [400000000, 200000000, 100000000]])
class SdeBroadwellJ2Test(SdeBaseTest):
    def __init__(self, mpitask, arraysize):
        super().__init__()
        ompthread = int(72/mpitask)
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.build_system.cppflags = [
            '-D_SDE',
            '-DSTREAM_ARRAY_SIZE=%s' % arraysize,
            '-DNTIMES=50'
        ]
        self.time_limit = (0, 10, 0)
        self.exclusive = True
        self.num_tasks = mpitask
        self.num_tasks_per_node = mpitask
        self.num_cpus_per_task = int(ompthread)
        self.num_tasks_per_core = 2
        self.use_multithreading = True
        self.name = 'sde_n.' + '{:010d}'.format(arraysize) + \
                    '_MPI.' + '{:03d}'.format(mpitask) + \
                    '_OpenMP.' + '{:03d}'.format(ompthread) + \
                    '_j.%s' % self.num_tasks_per_core
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

    @property
    @sn.sanity_function
    def arithmetic_intensity(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        byts = sn.extractsingle(r'^--->Total Bytes = (?P<byts>\d+)',
                                self.rpt, 'byts', int)
        return flops/byts

    @property
    @sn.sanity_function
    def gflops(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        sec = sn.extractsingle(r'^Triad:\s+\d+\.\d+\s+(?P<avgtime>\d+\.\d+)',
                               self.stdout, 'avgtime', float)
        step = sn.extractsingle(r'^Each kernel will be executed (?P<step>\d+)',
                                self.stdout, 'step', int)
        return ((flops/(sec*step))/10**9)
