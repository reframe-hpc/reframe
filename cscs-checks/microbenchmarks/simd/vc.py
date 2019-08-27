import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class VcSimdTest(rfm.RegressionTest):
    """
    Testing https://github.com/VcDevel/Vc
    Example performance report will look like:
    > reframe --system dom:mc -p PrgEnv-gnu -r -c vc.py --performance-report
        PERFORMANCE REPORT
        -----------------------------------------------------------------------
        VcSimdTest
        - dom:mc
           - PrgEnv-gnu
              * speedup: 1.3813
    > reframe --system dom:gpu -p PrgEnv-gnu -r -c vc.py --performance-report
        PERFORMANCE REPORT
        -----------------------------------------------------------------------
        VcSimdTest
        - dom:gpu
           - PrgEnv-gnu
              * speedup: 1.3824
    """
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.descr = 'finitediff example'
        self.build_system = 'SingleSource'
        self.testname = 'finitediff'
        src_url = 'https://raw.githubusercontent.com/VcDevel/Vc/1.4/examples'
        src1 = '%s/tsc.h' % src_url
        src2 = '%s/finitediff/main.cpp' % src_url
        self.prebuild_cmd = [
            'wget %s %s' % (src1, src2),
            'sed -ie "s-../tsc.h-./tsc.h-" main.cpp',
        ]
        self.sourcesdir = None
        self.sourcepath = 'main.cpp'
        self.executable = '%s.exe' % self.testname
        self.modules = ['Vc/1.4.1-CrayGNU-19.06']
        self.build_system.cxxflags = [
            '-DVc_IMPL=AVX2', '-std=c++14', '-O3', '-DNDEBUG', '-dynamic',
            '-I$EBROOTVC/include']
        self.build_system.ldflags = ['-L$EBROOTVC/lib', '-lVc']
        self.maintainers = ['JG']
        self.tags = {'benchmark', 'diagnostic'}
        self.exclusive = True
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
        }
        self.sanity_patterns = sn.assert_found('Speedup:', self.stdout)
        self.cycles = sn.extractall(r'^cycle count: (?P<cycles>\d+) \|',
                                    self.stdout, 'cycles', int)
        self.perf_patterns = {
            'speedup': self.speedup,
        }
        self.reference = {
            'dom:gpu': {
                'speedup': (1.38, -0.2, 0.2, '')
            },
            'dom:mc': {
                'speedup': (1.32, -0.2, 0.2, '')
            },
            '*': {
                'speedup': (1.0, None, None, '')
            }
        }

    @property
    @sn.sanity_function
    def speedup(self):
        # just showing how speedup is being calculated:
        sp = sn.round(self.cycles[0] / self.cycles[1], 3)
        return sp
