import os
import tempfile
import unittest

import reframe.core.runtime as rt
import reframe.utility.os_ext as os_ext
import reframe.utility.sanity as sn
import unittests.fixtures as fixtures
from reframe.core.exceptions import (BuildError, PipelineError, ReframeError,
                                     ReframeSyntaxError, SanityError)
from reframe.core.pipeline import (CompileOnlyRegressionTest, RegressionTest,
                                   RunOnlyRegressionTest)
from reframe.frontend.loader import RegressionCheckLoader


class TestRegressionTest(unittest.TestCase):
    def setup_local_execution(self):
        self.partition = rt.runtime().system.partition('login')
        self.progenv = self.partition.environment('builtin-gcc')

    def setup_remote_execution(self):
        self.partition = fixtures.partition_with_scheduler()
        if self.partition is None:
            self.skipTest('job submission not supported')

        try:
            self.progenv = self.partition.environs[0]
        except IndexError:
            self.skipTest('no environments configured for partition: %s' %
                          self.partition.fullname)

    def setUp(self):
        self.setup_local_execution()
        self.loader = RegressionCheckLoader(['unittests/resources/checks'])

        # Set runtime prefix
        rt.runtime().resources.prefix = tempfile.mkdtemp(dir='unittests')

    def tearDown(self):
        os_ext.rmtree(rt.runtime().resources.prefix)
        os_ext.rmtree('.rfm_testing', ignore_errors=True)

    def replace_prefix(self, filename, new_prefix):
        basename = os.path.basename(filename)
        return os.path.join(new_prefix, basename)

    def keep_files_list(self, test, compile_only=False):
        from reframe.core.deferrable import evaluate

        ret = [self.replace_prefix(evaluate(test.stdout), test.outputdir),
               self.replace_prefix(evaluate(test.stderr), test.outputdir)]

        if not compile_only:
            ret.append(self.replace_prefix(test.job.script_filename,
                                           test.outputdir))

        ret.extend([self.replace_prefix(f, test.outputdir)
                    for f in test.keep_files])
        return ret

    def test_environ_setup(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]
        test.modules = ['testmod_foo']
        test.variables = {'_FOO_': '1', '_BAR_': '2'}
        test.local = True

        test.setup(self.partition, self.progenv)

        for k in test.variables.keys():
            self.assertNotIn(k, os.environ)

        # Manually unload the environment
        self.progenv.unload()

    def _run_test(self, test, compile_only=False):
        test.setup(self.partition, self.progenv)
        test.compile()
        test.compile_wait()
        test.run()
        test.wait()
        test.check_sanity()
        test.check_performance()
        test.cleanup(remove_files=True)
        self.assertFalse(os.path.exists(test.stagedir))
        for f in self.keep_files_list(test, compile_only):
            self.assertTrue(os.path.exists(f))

    @fixtures.switch_to_user_runtime
    def test_hellocheck(self):
        self.setup_remote_execution()
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]
        self._run_test(test)

    @fixtures.switch_to_user_runtime
    def test_hellocheck_make(self):
        self.setup_remote_execution()
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck_make.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]
        self._run_test(test)

    def test_hellocheck_local(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]

        # Test also the prebuild/postbuild functionality
        test.prebuild_cmd  = ['touch prebuild']
        test.postbuild_cmd = ['touch postbuild']
        test.keepfiles = ['prebuild', 'postbuild']

        # Force local execution of the test
        test.local = True
        self._run_test(test)

    def test_hellocheck_local_prepost_run(self):
        @sn.sanity_function
        def stagedir(test):
            return test.stagedir

        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]

        # Test also the prebuild/postbuild functionality
        test.pre_run  = ['echo prerun: `pwd`']
        test.post_run = ['echo postrun: `pwd`']
        pre_run_path = sn.extractsingle(r'^prerun: (\S+)', test.stdout, 1)
        post_run_path = sn.extractsingle(r'^postrun: (\S+)', test.stdout, 1)
        test.sanity_patterns = sn.all([
            sn.assert_eq(stagedir(test), pre_run_path),
            sn.assert_eq(stagedir(test), post_run_path),
        ])

        # Force local execution of the test
        test.local = True
        self._run_test(test)

    def test_hellocheck_local_prepost_run_in_setup(self):
        def custom_setup(obj, partition, environ, **job_opts):
            super(obj.__class__, obj).setup(partition, environ, **job_opts)
            obj.pre_run  = ['echo Prerunning cmd from setup phase']
            obj.post_run = ['echo Postruning cmd from setup phase']

        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Monkey patch the setup method of the test
        test.setup = custom_setup.__get__(test)

        # Use test environment for the regression check
        test.valid_prog_environs = ['*']

        test.sanity_patterns = sn.all([
            sn.assert_found(r'^Prerunning cmd from setup phase', test.stdout),
            sn.assert_found(r'Hello, World\!', test.stdout),
            sn.assert_found(r'^Postruning cmd from setup phase', test.stdout)
        ])

        # Force local execution of the test
        test.local = True
        self._run_test(test)

    def test_run_only_sanity(self):
        test = RunOnlyRegressionTest('runonlycheck',
                                     'unittests/resources/checks')
        test.executable = './hello.sh'
        test.executable_opts = ['Hello, World!']
        test.local = True
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.sanity_patterns = sn.assert_found(r'Hello, World\!', test.stdout)
        self._run_test(test)

    def test_compile_only_failure(self):
        test = CompileOnlyRegressionTest('compileonlycheck',
                                         'unittests/resources/checks')
        test.sourcepath = 'compiler_failure.c'
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.setup(self.partition, self.progenv)
        test.compile()
        self.assertRaises(BuildError, test.compile_wait)

    def test_compile_only_warning(self):
        test = CompileOnlyRegressionTest('compileonlycheckwarning',
                                         'unittests/resources/checks')
        test.build_system = 'SingleSource'
        test.build_system.srcfile = 'compiler_warning.c'
        test.build_system.cflags = ['-Wall']
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.sanity_patterns = sn.assert_found(r'warning', test.stderr)
        self._run_test(test, compile_only=True)

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_supports_system(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        test.valid_systems = ['*']
        self.assertTrue(test.supports_system('gpu'))
        self.assertTrue(test.supports_system('login'))
        self.assertTrue(test.supports_system('testsys:gpu'))
        self.assertTrue(test.supports_system('testsys:login'))

        test.valid_systems = ['testsys']
        self.assertTrue(test.supports_system('gpu'))
        self.assertTrue(test.supports_system('login'))
        self.assertTrue(test.supports_system('testsys:gpu'))
        self.assertTrue(test.supports_system('testsys:login'))

        test.valid_systems = ['testsys:gpu']
        self.assertTrue(test.supports_system('gpu'))
        self.assertFalse(test.supports_system('login'))
        self.assertTrue(test.supports_system('testsys:gpu'))
        self.assertFalse(test.supports_system('testsys:login'))

        test.valid_systems = ['testsys:login']
        self.assertFalse(test.supports_system('gpu'))
        self.assertTrue(test.supports_system('login'))
        self.assertFalse(test.supports_system('testsys:gpu'))
        self.assertTrue(test.supports_system('testsys:login'))

        test.valid_systems = ['foo']
        self.assertFalse(test.supports_system('gpu'))
        self.assertFalse(test.supports_system('login'))
        self.assertFalse(test.supports_system('testsys:gpu'))
        self.assertFalse(test.supports_system('testsys:login'))

    def test_supports_environ(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        test.valid_prog_environs = ['*']
        self.assertTrue(test.supports_environ('foo1'))
        self.assertTrue(test.supports_environ('foo-env'))
        self.assertTrue(test.supports_environ('*'))

        test.valid_prog_environs = ['PrgEnv-foo-*']
        self.assertTrue(test.supports_environ('PrgEnv-foo-version1'))
        self.assertTrue(test.supports_environ('PrgEnv-foo-version2'))
        self.assertFalse(test.supports_environ('PrgEnv-boo-version1'))
        self.assertFalse(test.supports_environ('Prgenv-foo-version1'))

    def test_sourcesdir_none(self):
        test = RegressionTest('hellocheck', 'unittests/resources/checks')
        test.sourcesdir = None
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        self.assertRaises(ReframeError, self._run_test, test)

    def test_sourcesdir_none_generated_sources(self):
        test = RegressionTest('hellocheck_generated_sources',
                              'unittests/resources/checks')
        test.sourcesdir = None
        test.prebuild_cmd = ["printf '#include <stdio.h>\\n int main(){ "
                             "printf(\"Hello, World!\\\\n\"); return 0; }' "
                             "> hello.c"]
        test.executable = './hello'
        test.sourcepath = 'hello.c'
        test.local = True
        test.valid_systems = ['*']
        test.valid_prog_environs = ['*']
        test.sanity_patterns = sn.assert_found(r'Hello, World\!', test.stdout)
        self._run_test(test)

    def test_sourcesdir_none_compile_only(self):
        test = CompileOnlyRegressionTest('hellocheck',
                                         'unittests/resources/checks')
        test.sourcesdir = None
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        self.assertRaises(BuildError, self._run_test, test)

    def test_sourcesdir_none_run_only(self):
        test = RunOnlyRegressionTest('hellocheck',
                                     'unittests/resources/checks')
        test.sourcesdir = None
        test.executable = 'echo'
        test.executable_opts = ["Hello, World!"]
        test.local = True
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.sanity_patterns = sn.assert_found(r'Hello, World\!', test.stdout)
        self._run_test(test)

    def test_sourcepath_abs(self):
        test = CompileOnlyRegressionTest('compileonlycheck',
                                         'unittests/resources/checks')
        test.valid_prog_environs = [self.progenv.name]
        test.valid_systems = ['*']
        test.setup(self.partition, self.progenv)
        test.sourcepath = '/usr/src'
        self.assertRaises(PipelineError, test.compile)

    def test_sourcepath_upref(self):
        test = CompileOnlyRegressionTest('compileonlycheck',
                                         'unittests/resources/checks')
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.setup(self.partition, self.progenv)
        test.sourcepath = '../hellosrc'
        self.assertRaises(PipelineError, test.compile)

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_extra_resources(self):
        # Load test site configuration
        test = RegressionTest('dummycheck', 'unittests/resources/checks')
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.extra_resources = {
            'gpu': {'num_gpus_per_node': 2},
            'datawarp': {'capacity': '100GB', 'stagein_src': '/foo'}
        }
        partition = rt.runtime().system.partition('gpu')
        environ = partition.environment('builtin-gcc')
        test.setup(partition, environ)
        test.job.options += ['--foo']
        expected_job_options = ['--gres=gpu:2',
                                '#DW jobdw capacity=100GB',
                                '#DW stage_in source=/foo',
                                '--foo']
        self.assertCountEqual(expected_job_options, test.job.options)


class TestNewStyleChecks(unittest.TestCase):
    def test_regression_test(self):
        class MyTest(RegressionTest):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

        test = MyTest(1, 2)
        self.assertEqual(os.path.abspath(os.path.dirname(__file__)),
                         test.prefix)
        self.assertEqual('TestNewStyleChecks.test_regression_test.'
                         '<locals>.MyTest_1_2', test.name)

    def test_regression_test_strange_names(self):
        class C:
            def __init__(self, a):
                self.a = a

            def __repr__(self):
                return 'C(%s)' % self.a

        class MyTest(RegressionTest):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

        test = MyTest('(a*b+c)/12', C(33))
        self.assertEqual(
            'TestNewStyleChecks.test_regression_test_strange_names.'
            '<locals>.MyTest__a_b_c__12_C_33_', test.name)

    def test_user_inheritance(self):
        class MyBaseTest(RegressionTest):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

        class MyTest(MyBaseTest):
            def __init__(self):
                super().__init__(1, 2)

        test = MyTest()
        self.assertEqual('TestNewStyleChecks.test_user_inheritance.'
                         '<locals>.MyTest', test.name)

    def test_runonly_test(self):
        class MyTest(RunOnlyRegressionTest):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

        test = MyTest(1, 2)
        self.assertEqual(os.path.abspath(os.path.dirname(__file__)),
                         test.prefix)
        self.assertEqual('TestNewStyleChecks.test_runonly_test.'
                         '<locals>.MyTest_1_2', test.name)

    def test_compileonly_test(self):
        class MyTest(CompileOnlyRegressionTest):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

        test = MyTest(1, 2)
        self.assertEqual(os.path.abspath(os.path.dirname(__file__)),
                         test.prefix)
        self.assertEqual('TestNewStyleChecks.test_compileonly_test.'
                         '<locals>.MyTest_1_2', test.name)

    def test_registration(self):
        import sys
        import unittests.resources.checks_unlisted.good as mod
        checks = mod._rfm_gettests()
        self.assertEqual(13, len(checks))
        self.assertEqual([mod.MyBaseTest(0, 0),
                          mod.MyBaseTest(0, 1),
                          mod.MyBaseTest(1, 0),
                          mod.MyBaseTest(1, 1),
                          mod.MyBaseTest(2, 0),
                          mod.MyBaseTest(2, 1),
                          mod.AnotherBaseTest(0, 0),
                          mod.AnotherBaseTest(0, 1),
                          mod.AnotherBaseTest(1, 0),
                          mod.AnotherBaseTest(1, 1),
                          mod.AnotherBaseTest(2, 0),
                          mod.AnotherBaseTest(2, 1),
                          mod.MyBaseTest(10, 20)], checks)


class TestSanityPatterns(unittest.TestCase):
    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def setUp(self):
        # Set up the test runtime
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        rt.runtime().resources.prefix = self.resourcesdir

        # Set up RegressionTest instance
        self.test = RegressionTest('test_performance',
                                   'unittests/resources/checks')
        self.partition = rt.runtime().system.partition('gpu')
        self.progenv = self.partition.environment('builtin-gcc')

        self.test.setup(self.partition, self.progenv)
        self.test.reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1),
                'value2': (1.7, -0.1, 0.1),
            },
            'testsys:gpu': {
                'value3': (3.1, -0.1, 0.1),
            }
        }

        self.perf_file = tempfile.NamedTemporaryFile(mode='wt', delete=False)
        self.output_file = tempfile.NamedTemporaryFile(mode='wt', delete=False)
        self.test.perf_patterns = {
            'value1': sn.extractsingle(r'performance1 = (\S+)',
                                       self.perf_file.name, 1, float),
            'value2': sn.extractsingle(r'performance2 = (\S+)',
                                       self.perf_file.name, 1, float),
            'value3': sn.extractsingle(r'performance3 = (\S+)',
                                       self.perf_file.name, 1, float)
        }

        self.test.sanity_patterns = sn.assert_found(r'result = success',
                                                    self.output_file.name)

    def tearDown(self):
        self.perf_file.close()
        self.output_file.close()
        os.remove(self.perf_file.name)
        os.remove(self.output_file.name)
        os_ext.rmtree(self.resourcesdir)

    def write_performance_output(self, fp=None, **kwargs):
        if not fp:
            fp = self.perf_file

        for k, v in kwargs.items():
            fp.write('%s = %s\n' % (k, v))

        fp.close()

    def test_success(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.output_file.write('result = success\n')
        self.output_file.close()
        self.test.check_sanity()
        self.test.check_performance()

    def test_sanity_failure(self):
        self.output_file.write('result = failure\n')
        self.output_file.close()
        self.assertRaises(SanityError, self.test.check_sanity)

    def test_sanity_failure_noassert(self):
        self.test.sanity_patterns = sn.findall(r'result = success',
                                               self.output_file.name)
        self.output_file.write('result = failure\n')
        self.output_file.close()
        self.assertRaises(SanityError, self.test.check_sanity)

    def test_sanity_multiple_patterns(self):
        self.output_file.write('result1 = success\n')
        self.output_file.write('result2 = success\n')
        self.output_file.close()

        # Simulate a pure sanity test; invalidate the reference values
        self.test.reference = {}
        self.test.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'result\d = success', self.output_file.name)),
            2)
        self.test.check_sanity()

        # Require more patterns to be present
        self.test.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'result\d = success', self.output_file.name)),
            3)
        self.assertRaises(SanityError, self.test.check_sanity)

    def test_sanity_multiple_files(self):
        files = [tempfile.NamedTemporaryFile(mode='wt', prefix='regtmp',
                                             dir=self.test.stagedir,
                                             delete=False)
                 for i in range(2)]

        for f in files:
            f.write('result = success\n')
            f.close()

        self.test.sanity_patterns = sn.all([
            sn.assert_found(r'result = success', files[0].name),
            sn.assert_found(r'result = success', files[1].name)
        ])
        self.test.check_sanity()
        for f in files:
            os.remove(f.name)

    def test_performance_failure(self):
        self.write_performance_output(performance1=1.0,
                                      performance2=1.8,
                                      performance3=3.3)
        self.output_file.write('result = success\n')
        self.output_file.close()
        self.test.check_sanity()
        self.assertRaises(SanityError, self.test.check_performance)

    def test_unknown_tag(self):
        self.test.reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1),
                'value2': (1.7, -0.1, 0.1),
                'foo': (3.1, -0.1, 0.1),
            }
        }

        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.assertRaises(SanityError, self.test.check_performance)

    def test_unknown_system(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {
            'testsys:login': {
                'value1': (1.4, -0.1, 0.1),
                'value2': (1.7, -0.1, 0.1),
                'value3': (3.1, -0.1, 0.1),
            }
        }
        self.assertRaises(SanityError, self.test.check_performance)

    def test_default_reference(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {
            '*': {
                'value1': (1.4, -0.1, 0.1),
                'value2': (1.7, -0.1, 0.1),
                'value3': (3.1, -0.1, 0.1),
            }
        }

        self.test.check_performance()

    def test_tag_resolution(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1),
                'value2': (1.7, -0.1, 0.1),
            },
            '*': {
                'value3': (3.1, -0.1, 0.1),
            }
        }
        self.test.check_performance()

    def test_perf_var_evaluation(self):
        # All performance values must be evaluated, despite the first one
        # failing To test this, we need an extract function that will have a
        # side effect when evaluated, whose result we will check after calling
        # `check_performance()`.
        logfile = 'perf.log'

        @sn.sanity_function
        def extract_perf(patt, tag):
            val = sn.evaluate(
                sn.extractsingle(patt, self.perf_file.name, tag, float))

            with open('perf.log', 'a') as fp:
                fp.write('%s=%s' % (tag, val))

            return val

        self.test.perf_patterns = {
            'value1': extract_perf(r'performance1 = (?P<v1>\S+)', 'v1'),
            'value2': extract_perf(r'performance2 = (?P<v2>\S+)', 'v2'),
            'value3': extract_perf(r'performance3 = (?P<v3>\S+)', 'v3')
        }
        self.write_performance_output(performance1=1.0,
                                      performance2=1.8,
                                      performance3=3.3)
        with self.assertRaises(SanityError) as cm:
            self.test.check_performance()

        logfile = os.path.join(self.test.stagedir, logfile)
        with open(logfile) as fp:
            log_output = fp.read()

        self.assertIn('v1', log_output)
        self.assertIn('v2', log_output)
        self.assertIn('v3', log_output)
