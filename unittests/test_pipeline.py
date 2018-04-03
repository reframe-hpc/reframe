import os
import shutil
import tempfile
import unittest

import reframe.utility.sanity as sn
import unittests.fixtures as fixtures
from reframe.core.exceptions import (ReframeError, PipelineError, SanityError,
                                     CompilationError)
from reframe.core.modules import get_modules_system
from reframe.core.pipeline import (CompileOnlyRegressionTest, RegressionTest,
                                   RunOnlyRegressionTest)
from reframe.frontend.loader import RegressionCheckLoader
from reframe.frontend.resources import ResourcesManager


class TestRegressionTest(unittest.TestCase):
    def setUp(self):
        get_modules_system().searchpath_add(fixtures.TEST_MODULES)

        # Load a system configuration
        self.system, self.partition, self.progenv = fixtures.get_test_config()
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.loader = RegressionCheckLoader(['unittests/resources'])
        self.resources = ResourcesManager(prefix=self.resourcesdir)

    def tearDown(self):
        shutil.rmtree(self.resourcesdir, ignore_errors=True)

    def setup_from_site(self):
        self.partition = fixtures.partition_with_scheduler(None)

        # pick the first environment of partition
        if self.partition.environs:
            self.progenv = self.partition.environs[0]

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
            'unittests/resources/hellocheck.py',
            system=self.system, resources=self.resources
        )[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]
        test.modules = ['testmod_foo']
        test.variables = {'_FOO_': '1', '_BAR_': '2'}
        test.local = True

        test.setup(self.partition, self.progenv)
        for m in test.modules:
            self.assertTrue(get_modules_system().is_module_loaded(m))

        for k, v in test.variables.items():
            self.assertEqual(os.environ[k], v)

        # Manually unload the environment
        self.progenv.unload()

    def _run_test(self, test, compile_only=False):
        test.setup(self.partition, self.progenv)
        test.compile()
        test.run()
        test.wait()
        test.check_sanity()
        test.check_performance()
        test.cleanup(remove_files=True)
        self.assertFalse(os.path.exists(test.stagedir))
        for f in self.keep_files_list(test, compile_only):
            self.assertTrue(os.path.exists(f))

    @unittest.skipIf(not fixtures.partition_with_scheduler(None),
                     'job submission not supported')
    def test_hellocheck(self):
        self.setup_from_site()
        test = self.loader.load_from_file(
            'unittests/resources/hellocheck.py',
            system=self.system, resources=self.resources
        )[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]
        self._run_test(test)

    @unittest.skipIf(not fixtures.partition_with_scheduler(None),
                     'job submission not supported')
    def test_hellocheck_make(self):
        self.setup_from_site()
        test = self.loader.load_from_file(
            'unittests/resources/hellocheck_make.py',
            system=self.system, resources=self.resources
        )[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]
        self._run_test(test)

    def test_hellocheck_local(self):
        test = self.loader.load_from_file(
            'unittests/resources/hellocheck.py',
            system=self.system, resources=self.resources
        )[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.progenv.name]

        # Test also the prebuild/postbuild functionality
        test.prebuild_cmd  = ['touch prebuild']
        test.postbuild_cmd = ['touch postbuild']
        test.keepfiles = ['prebuild', 'postbuild']

        # Force local execution of the test
        test.local = True
        self._run_test(test)

    def test_hellocheck_local_slashes(self):
        # Try to fool path creation by adding slashes to environment partitions
        # names
        from reframe.core.environments import ProgEnvironment

        self.progenv = ProgEnvironment('bad/name', self.progenv.modules,
                                       self.progenv.variables)

        # That's a bit hacky, but we are in a unit test
        self.system._name += os.sep + 'bad'
        self.partition._name += os.sep + 'bad'
        self.test_hellocheck_local()

    def test_hellocheck_local_prepost_run(self):
        @sn.sanity_function
        def stagedir(test):
            return test.stagedir

        test = self.loader.load_from_file(
            'unittests/resources/hellocheck.py',
            system=self.system, resources=self.resources
        )[0]

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

    def test_run_only_sanity(self):
        test = RunOnlyRegressionTest('runonlycheck',
                                     'unittests/resources',
                                     resources=self.resources,
                                     system=self.system)
        test.executable = './hello.sh'
        test.executable_opts = ['Hello, World!']
        test.local = True
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.sanity_patterns = sn.assert_found(r'Hello, World\!', test.stdout)
        self._run_test(test)

    def test_compile_only_failure(self):
        test = CompileOnlyRegressionTest('compileonlycheck',
                                         'unittests/resources',
                                         resources=self.resources,
                                         system=self.system)
        test.sourcepath = 'compiler_failure.c'
        test.valid_prog_environs = [self.progenv.name]
        test.valid_systems = [self.system.name]
        test.setup(self.partition, self.progenv)
        self.assertRaises(CompilationError, test.compile)

    def test_compile_only_warning(self):
        test = CompileOnlyRegressionTest('compileonlycheckwarning',
                                         'unittests/resources',
                                         resources=self.resources,
                                         system=self.system)
        test.sourcepath = 'compiler_warning.c'
        self.progenv.cflags = '-Wall'
        test.valid_prog_environs = [self.progenv.name]
        test.valid_systems = [self.system.name]
        test.sanity_patterns = sn.assert_found(r'warning', test.stderr)
        self._run_test(test, compile_only=True)

    def test_supports_system(self):
        test = self.loader.load_from_file(
            'unittests/resources/hellocheck.py',
            system=self.system, resources=self.resources
        )[0]

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
            'unittests/resources/hellocheck.py',
            system=self.system, resources=self.resources
        )[0]

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
        test = RegressionTest('hellocheck',
                              'unittests/resources',
                              resources=self.resources,
                              system=self.system)
        test.sourcesdir = None
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        self.assertRaises(ReframeError, self._run_test, test)

    def test_sourcesdir_none_generated_sources(self):
        test = RegressionTest('hellocheck_generated_sources',
                              'unittests/resources',
                              resources=self.resources,
                              system=self.system)
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
                                         'unittests/resources',
                                         resources=self.resources,
                                         system=self.system)
        test.sourcesdir = None
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        self.assertRaises(CompilationError, self._run_test, test)

    def test_sourcesdir_none_run_only(self):
        test = RunOnlyRegressionTest('hellocheck',
                                     'unittests/resources',
                                     resources=self.resources,
                                     system=self.system)
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
                                         'unittests/resources',
                                         resources=self.resources,
                                         system=self.system)
        test.valid_prog_environs = [self.progenv.name]
        test.valid_systems = [self.system.name]
        test.setup(self.partition, self.progenv)
        test.sourcepath = '/usr/src'
        self.assertRaises(PipelineError, test.compile)

    def test_sourcepath_upref(self):
        test = CompileOnlyRegressionTest('compileonlycheck',
                                         'unittests/resources',
                                         resources=self.resources,
                                         system=self.system)
        test.valid_prog_environs = [self.progenv.name]
        test.valid_systems = [self.system.name]
        test.setup(self.partition, self.progenv)
        test.sourcepath = '../hellosrc'
        self.assertRaises(PipelineError, test.compile)

    def test_extra_resources(self):
        # Load test site configuration
        system, partition, progenv = fixtures.get_test_config()
        test = RegressionTest('dummycheck', 'unittests/resources',
                              resources=self.resources, system=self.system)
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.extra_resources = {
            'gpu': {'num_gpus_per_node': 2},
            'datawarp': {'capacity': '100GB', 'stagein_src': '/foo'}
        }
        test.setup(self.partition, self.progenv)
        test.job.options += ['--foo']
        expected_job_options = ['--gres=gpu:2',
                                '#DW jobdw capacity=100GB',
                                '#DW stage_in source=/foo',
                                '--foo']
        self.assertCountEqual(expected_job_options, test.job.options)


class TestSanityPatterns(unittest.TestCase):
    def setUp(self):
        # Load test site configuration
        self.system, self.partition, self.progenv = fixtures.get_test_config()

        # Set up RegressionTest instance
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.resources = ResourcesManager(prefix=self.resourcesdir)
        self.test = RegressionTest('test_performance',
                                   'unittests/resources',
                                   resources=self.resources,
                                   system=self.system)

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
        shutil.rmtree(self.resourcesdir)

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
