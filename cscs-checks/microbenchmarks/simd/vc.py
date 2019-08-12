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
              * speedup: 1.3813 cyc
    > reframe --system dom:gpu -p PrgEnv-gnu -r -c vc.py --performance-report
        PERFORMANCE REPORT
        -----------------------------------------------------------------------
        VcSimdTest
        - dom:gpu
           - PrgEnv-gnu
              * speedup: 1.3824 cyc
    """
    def __init__(self):
        super().__init__()
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.descr = 'finitediff example'
        self.build_system = 'SingleSource'
        self.testname = 'finitediff'
        self.sourcepath = 'Vc/%s/main.cpp' % self.testname
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
        regex_cyc = (r'^cycle count: (?P<cycles>\d+) \|')
        self.cyc1 = sn.extractall(regex_cyc, self.stdout, 'cycles', int)[0]
        self.cyc2 = sn.extractall(regex_cyc, self.stdout, 'cycles', int)[1]
        self.perf_patterns = {
            'speedup': self.speedup,
        }
        self.reference = {
            'dom:gpu': {
                'speedup': (1.38, -0.2, 0.2, 'cyc')
            },
            'dom:mc': {
                'speedup': (1.32, -0.2, 0.2, 'cyc')
            },
            '*': {
                'speedup': (1.0, None, None, 'cyc')
            }
        }

    @property
    @sn.sanity_function
    def speedup(self):
        sp = float(str(self.cyc1 / self.cyc2)[0:6])
        # sp = "{:6.4f}".format(self.cyc1 / self.cyc2)
        return sp

