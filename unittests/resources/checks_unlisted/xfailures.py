import reframe as rfm
import reframe.core.builtins as builtins
import reframe.utility.sanity as sn


@rfm.simple_test
@rfm.xfail('xfail sanity', lambda test: (
    test.phase == 'sanity' and (test.status == 'xfail' or test.status == 'xpass')
))
class XfailTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo hello && echo perf=10 && echo time=0.1'
    reference = {
        '*': {
            'perf': (10, 0, 0, 'GB/s'),
            'time': (0.1, 0, 0, 's')
        }
    }
    status = parameter(['pass', 'fail', 'xfail', 'xpass'])
    phase  = parameter(['sanity', 'performance'])

    @sanity_function
    def validate(self):
        if self.phase == 'sanity' and self.status in ('fail', 'xfail'):
            return sn.assert_found(r'helo', self.stdout)
        else:
            return sn.assert_found(r'hello', self.stdout)

    @run_after('init')
    def set_references(self):
        if self.phase == 'sanity':
            return

        if self.status == 'fail':
            self.reference['*:perf'] = (9, 0, 0, 'GB/s')
        elif self.status == 'xfail':
            self.reference['*:perf'] = builtins.xfail('xfail perf', (9, 0, 0, 'GB/s'))
        elif self.status == 'xpass':
            self.reference['*:perf'] = builtins.xfail('xfail perf', self.reference['*:perf'])

    @performance_function('GB/s')
    def perf(self):
        return sn.extractsingle(r'perf=(\S+)', self.stdout, 1, float)

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'time=(\S+)', self.stdout, 1, float)
