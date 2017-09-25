import os
import shutil
import stat
import tempfile
import unittest

import reframe.settings as settings

from reframe.core.pipeline import *
from reframe.core.exceptions import ReframeError, CompilationError
from reframe.core.modules import *
from reframe.frontend.loader import *
from reframe.frontend.resources import ResourcesManager
from reframe.utility.functions import standard_threshold

from unittests.fixtures import TEST_MODULES, TEST_SITE_CONFIG
from unittests.fixtures import system_with_scheduler


class TestRegression(unittest.TestCase):
    def setUp(self):
        module_path_add([TEST_MODULES])

        # Load a system configuration
        self.site_config = SiteConfiguration()
        self.site_config.load_from_dict(TEST_SITE_CONFIG)
        self.system    = self.site_config.systems['testsys']
        self.partition = self.system.partition('gpu')
        self.progenv   = self.partition.environment('builtin-gcc')

        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.loader    = RegressionCheckLoader(['unittests/resources'])
        self.resources = ResourcesManager(prefix=self.resourcesdir)

    def tearDown(self):
        shutil.rmtree(self.resourcesdir, ignore_errors=True)

    def setup_from_site(self):
        self.partition = system_with_scheduler(None)

        # pick the first environment of partition
        if self.partition.environs:
            self.progenv = self.partition.environs[0]

    def replace_prefix(self, filename, new_prefix):
        basename = os.path.basename(filename)
        return os.path.join(new_prefix, basename)

    def keep_files_list(self, test, compile_only=False):
        ret = [self.replace_prefix(test.stdout, test.outputdir),
               self.replace_prefix(test.stderr, test.outputdir)]

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
            self.assertTrue(module_present(m))

        for k, v in test.variables.items():
            self.assertEqual(os.environ[k], v)

        # Manually unload the environment
        self.progenv.unload()

    def _run_test(self, test, compile_only=False, performance_result=True):
        test.setup(self.partition, self.progenv)
        test.compile()
        test.run()
        test.wait()
        self.assertTrue(test.check_sanity())
        self.assertEqual(test.check_performance(), performance_result)
        test.cleanup(remove_files=True)
        self.assertFalse(os.path.exists(test.stagedir))
        for f in self.keep_files_list(test, compile_only):
            self.assertTrue(os.path.exists(f))

    @unittest.skipIf(not system_with_scheduler(None),
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

    @unittest.skipIf(not system_with_scheduler(None),
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
        self.system.name    += os.sep + 'bad'
        self.progenv.name   += os.sep + 'bad'
        self.partition.name += os.sep + 'bad'
        self.test_hellocheck_local()

    def test_run_only(self):
        test = RunOnlyRegressionTest('runonlycheck',
                                     'unittests/resources',
                                     resources=self.resources,
                                     system=self.system)
        test.executable = './hello.sh'
        test.executable_opts = ['Hello, World!']
        test.local = True
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        test.sanity_patterns = {
            '-' : {'Hello, World\!': []}
        }
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
        test.sanity_patterns = {
            '&2': {'warning': []}
        }
        self._run_test(test, compile_only=True)

    def test_supports_system(self):
        test = self.loader.load_from_file(
            'unittests/resources/hellocheck.py',
            system=self.system, resources=self.resources
        )[0]
        test.current_system = System('testsys')

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

    def test_sourcesdir_none(self):
        test = RegressionTest('hellocheck',
                              'unittests/resources',
                              resources=self.resources,
                              system=self.system)
        test.sourcesdir = None
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        self.assertRaises(ReframeError, self._run_test, test)

    def test_sourcesdir_none_compile_only(self):
        test = CompileOnlyRegressionTest('hellocheck',
                                         'unittests/resources',
                                         resources=self.resources,
                                         system=self.system)
        test.sourcesdir = None
        test.valid_prog_environs = ['*']
        test.valid_systems = ['*']
        self.assertRaises(ReframeError, self._run_test, test)

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
        test.sanity_patterns = {
            '-' : {'Hello, World\!': []}
        }
        self._run_test(test)


class TestRegressionOutputScan(unittest.TestCase):
    def setUp(self):
        self.system = System('testsys')
        self.system.partitions.append(SystemPartition('gpu', self.system))

        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.resources = ResourcesManager(prefix=self.resourcesdir)
        self.test = RegressionTest('test_performance',
                                   'unittests/resources',
                                   resources=self.resources,
                                   system=self.system)

        self.test.current_system    = self.system
        self.test.current_partition = self.system.partition('gpu')
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
            self.perf_file.name: {
                'performance1 = (?P<value1>\S+)': [
                    ('value1', float, standard_threshold)
                ],
                'performance2 = (?P<value2>\S+)': [
                    ('value2', float, standard_threshold)
                ],
                'performance3 = (?P<value3>\S+)': [
                    ('value3', float, standard_threshold)
                ]
            }
        }

        self.test.sanity_patterns = {
            self.output_file.name: {
                'result = success': []
            }
        }
        self.test.stagedir = self.test.prefix

    def tearDown(self):
        self.perf_file.close()
        self.output_file.close()
        os.remove(self.perf_file.name)
        os.remove(self.output_file.name)
        shutil.rmtree(self.resourcesdir)

    def write_performance_output(self, file=None, **kwargs):
        if not file:
            file = self.perf_file

        for k, v in kwargs.items():
            file.write('%s = %s\n' % (k, v))

        file.close()

    def custom_sanity(self, value, reference, **kwargs):
        return value == 'success'

    # custom threshold function
    def custom_threshold(self, value, reference, **kwargs):
        return value >= reference * 0.9 and value <= reference * 1.1

    def assertReportGeneration(self):
        # Assert that the different reports are generated without unexpected
        # exceptions; no check is made as of their contents
        self.test.sanity_info.scan_report()
        self.test.perf_info.scan_report()
        self.test.sanity_info.failure_report()
        self.test.perf_info.failure_report()

    def test_success(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.output_file.write('result = success\n')
        self.output_file.close()
        self.assertTrue(self.test.check_sanity())
        self.assertTrue(self.test.check_performance())

        # Verify that the sanity/perf. check info is collected correctly
        self.assertIsNotNone(self.test.sanity_info.matched_pattern(
            self.output_file.name, 'result = success'))

        expected_perf_info = {
            self.perf_file.name: {
                'performance1 = (?P<value1>\S+)': [
                    ('value1', 1.3, (1.4, -0.1, 0.1), True)
                ],
                'performance2 = (?P<value2>\S+)': [
                    ('value2', 1.8, (1.7, -0.1, 0.1), True)
                ],
                'performance3 = (?P<value3>\S+)': [
                    ('value3', 3.3, (3.1, -0.1, 0.1), True)
                ]
            }
        }
        for path, patterns in expected_perf_info.items():
            for patt, taglist in patterns.items():
                self.assertIsNotNone(
                    self.test.perf_info.matched_pattern(path, patt))

                for t in taglist:
                    tinfo = self.test.perf_info.matched_tag(path, patt, t[0])
                    self.assertIsNotNone(tinfo)
                    self.assertEquals(t, tinfo)

    def test_empty_file(self):
        self.output_file.close()
        self.test.sanity_patterns = {
            self.output_file.name : {'.*': []}
        }
        self.assertFalse(self.test.check_sanity())
        self.assertIsNone(self.test.sanity_info.matched_pattern(
            self.output_file.name, '.*'))

    def test_sanity_failure(self):
        self.output_file.write('result = failure\n')
        self.output_file.close()
        self.assertFalse(self.test.check_sanity())
        self.assertIsNone(self.test.sanity_info.matched_pattern(
            self.output_file.name, 'result = success'))

    def test_sanity_multiple_patterns(self):
        self.output_file.write('result1 = success\n')
        self.output_file.write('result2 = success\n')
        self.output_file.close()

        # Simulate a pure sanity test; invalidate the reference values
        self.test.reference = {}
        self.test.sanity_patterns = {
            self.output_file.name: {
                'result1 = success': [],
                'result2 = success': []
            }
        }
        self.assertTrue(self.test.check_sanity())

        # Require more patterns to be present
        self.test.sanity_patterns = {
            self.output_file.name: {
                'result1 = success': [],
                'result2 = success': [],
                'result3 = success': []
            }
        }
        self.assertFalse(self.test.check_sanity())
        self.assertIsNone(self.test.sanity_info.matched_pattern(
            self.output_file.name, 'result3 = success'))

    def test_multiple_files(self):
        # Create multiple files following the same pattern
        files = [tempfile.NamedTemporaryFile(mode='wt', prefix='regtmp',
                                             dir=self.test.prefix,
                                             delete=False)
                 for i in range(0, 2)]

        # Write the performance files
        for f in files:
            self.write_performance_output(f,
                                          performance1=1.3,
                                          performance2=1.8,
                                          performance3=3.3)

        # Reset the performance patterns; also put relative paths
        self.test.perf_patterns = {
            'regtmp*': {
                'performance1 = (?P<value1>\S+)': [
                    ('value1', float, standard_threshold)
                ],
                'performance2 = (?P<value2>\S+)': [
                    ('value2', float, standard_threshold)
                ],
                'performance3 = (?P<value3>\S+)': [
                    ('value3', float, standard_threshold)
                ],
            }
        }

        # Relative paths are resolved relative to stagedir
        self.assertTrue(self.test.check_performance())

        # Remove the performance files
        for f in files:
            os.remove(f.name)

    def test_invalid_conversion(self):
        self.write_performance_output(performance1='nodata',
                                      performance2=1.8,
                                      performance3=3.3)
        self.assertRaises(ReframeError, self.test.check_performance)

    def test_reference_file_not_found(self):
        self.output_file.write('result = success\n')
        self.output_file.close()

        # Remove read permissions
        os.chmod(self.output_file.name, stat.S_IWUSR)
        self.assertRaises(ReframeError, self.test.check_sanity)

    def test_below_threshold(self):
        self.write_performance_output(performance1=1.0,
                                      performance2=1.7,
                                      performance3=3.1)
        self.assertFalse(self.test.check_performance())

        # Verify collected match info
        tag, val, ref, res = self.test.perf_info.matched_tag(
            self.perf_file.name, 'performance1 = (?P<value1>\S+)', 'value1')
        self.assertFalse(res)

    def test_above_threshold(self):
        self.write_performance_output(performance1=1.4,
                                      performance2=2.7,
                                      performance3=3.2)
        self.assertFalse(self.test.check_performance())

        # Verify collected match info
        tag, val, ref, res = self.test.perf_info.matched_tag(
            self.perf_file.name, 'performance2 = (?P<value2>\S+)', 'value2')
        self.assertFalse(res)

    def test_invalid_threshold(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        # Invalidate lower threshold
        self.test.reference['testsys:value1'] = (1.4, -10, 0.1)
        self.assertRaises(ReframeError, self.test.check_performance)

        self.test.reference['testsys:value1'] = (1.4, 0.1, 0.1)
        self.assertRaises(ReframeError, self.test.check_performance)

        # Invalidate upper threshold
        self.test.reference['testsys:value1'] = (1.4, -0.1, 10)
        self.assertRaises(ReframeError, self.test.check_performance)

        self.test.reference['testsys:value1'] = (1.4, -0.1, -0.1)
        self.assertRaises(ReframeError, self.test.check_performance)

    def test_zero_reference(self):
        self.test.reference = {
            'testsys': {
                'value1': (0.0, -0.1, 0.1),
                'value2': (0.0, -0.1, 0.1),
                'value3': (0.0, -0.1, 0.1),
            }
        }

        self.write_performance_output(performance1=0.05,
                                      performance2=-0.05,
                                      performance3=0.0)
        self.assertTrue(self.test.check_performance())

    def test_zero_thresholds(self):
        self.test.reference = {
            'testsys': {
                'value1': (1.4, 0.0, 0.0),
                'value2': (1.7, 0.0, 0.0),
                'value3': (3.1, 0.0, 0.0),
            }
        }

        self.write_performance_output(performance1=1.41,
                                      performance2=1.69,
                                      performance3=3.11)
        self.assertFalse(self.test.check_performance())

    def test_unbounded(self):
        self.test.reference = {
            'testsys': {
                'value1': (1.4, None, None),
                'value2': (1.7, None, 0.1),
                'value3': (3.1, -0.1, None),
            }
        }

        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.assertTrue(self.test.check_performance())

    def test_no_threshold(self):
        self.test.reference = {
            'testsys': {
                'value1': (None, None, None),
                'value2': (1.7, None, 0.1),
                'value3': (3.1, -0.1, None),
            }
        }

        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.assertRaises(ReframeError, self.test.check_performance)

    def test_pattern_not_found(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      foo=3.3)
        self.assertFalse(self.test.check_performance())

    def test_custom_threshold(self):
        self.test.reference = {
            'testsys': {
                'value1': 1.4,
                'value2': 1.7,
                'value3': 3.1,
            }
        }

        self.test.perf_patterns = {
            self.perf_file.name: {
                'performance1 = (?P<value1>\S+)': [
                    ('value1', float, self.custom_threshold)
                ],
                'performance2 = (?P<value2>\S+)': [
                    ('value2', float, self.custom_threshold)
                ],
                'performance3 = (?P<value3>\S+)': [
                    ('value3', float,
                     lambda value, **kwargs: value >= 3.1)
                ],
            }
        }

        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.assertTrue(self.test.check_performance())

    def test_sanity_tags(self):
        self.test.reference = {}
        self.test.sanity_patterns = {
            self.output_file.name: {
                'result = (?P<result>\S+)': [
                    ('result', str, self.custom_sanity)
                ]
            }
        }

        self.output_file.write('result = success\n')
        self.output_file.close()
        self.assertTrue(self.test.check_sanity())

        self.output_file = open(self.output_file.name, 'wt')
        self.output_file.write('result = failure\n')
        self.output_file.close()
        self.assertFalse(self.test.check_sanity())

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
        self.assertRaises(ReframeError, self.test.check_performance)

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
        self.assertRaises(ReframeError, self.test.check_performance)

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

        self.assertTrue(self.test.check_performance())

    def test_tag_resolution(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)

        self.test.reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1),
                'value2': (1.7, -0.1, 0.1),
            },
            '*' : {
                'value3': (3.1, -0.1, 0.1),
            }
        }
        self.assertTrue(self.test.check_performance())

    def test_negative_threshold_success(self):
        self.write_performance_output(performance1=-1.3,
                                      performance2=-1.8,
                                      performance3=-3.3)

        self.test.reference = {
            '*': {
                'value1': (-1.4, -0.1, 0.1),
                'value2': (-1.7, -0.1, 0.1),
                'value3': (-3.1, -0.1, 0.1),
            }
        }

        self.assertTrue(self.test.check_performance())

    def test_negative_threshold_failure(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {
            '*': {
                'value1': (-1.4, -0.1, 0.1),
                'value2': (-1.7, -0.1, 0.1),
                'value3': (-3.1, -0.1, 0.1),
            }
        }
        self.assertFalse(self.test.check_performance())

    def test_negative_threshold_positive_ref(self):
        self.write_performance_output(performance1=-1.3,
                                      performance2=-1.8,
                                      performance3=-3.3)
        self.test.reference = {
            '*': {
                'value1': (1.4, -0.1, 0.1),
                'value2': (1.7, -0.1, 0.1),
                'value3': (3.1, -0.1, 0.1),
            }
        }
        self.assertFalse(self.test.check_performance())

    def test_eof_handler(self):
        self.output_file.write('result = success\n')
        self.output_file.write('result = success\n')
        self.output_file.write('result = success\n')
        self.output_file.write('\\e = success\n')
        self.output_file.close()

        class Parser:
            def __init__(self):
                self.count = 0

            def match_line(self, value, reference, **kwargs):
                self.count += 1
                return True

            def match_eof(self, **kwargs):
                return self.count == 3

        p = Parser()
        self.test.sanity_patterns = {
            self.output_file.name: {
                '(?P<success_string>result = success)': [
                    ('success_string', str, p.match_line)
                ],

                r'\\e = success': [],

                '\e': p.match_eof
            },
        }

        self.assertTrue(self.test.check_sanity())
        self.assertIn('\e',
                      self.test.sanity_patterns[self.output_file.name].keys())
        self.assertReportGeneration()

    def test_eof_handler_restore_on_failure(self):
        self.output_file.write('result = success\n')
        self.output_file.write('result = success\n')
        self.output_file.close()

        class Parser:
            def __init__(self):
                self.count = 0

            def match_line(self, value, reference, **kwargs):
                self.count += 1
                return True

            def match_eof(self, **kwargs):
                return self.count == 3

        p = Parser()
        self.test.sanity_patterns = {
            self.output_file.name: {
                '(?P<success_string>result = success)': [
                    ('success_string', str, p.match_line)
                ],
                '\e': p.match_eof
            },
        }

        self.assertFalse(self.test.check_sanity())
        self.assertIn('\e',
                      self.test.sanity_patterns[self.output_file.name].keys())
        self.assertReportGeneration()

    def test_patterns_empty(self):
        self.test.perf_patterns = {}
        self.test.sanity_patterns = {}
        self.assertTrue(self.test.check_sanity())
        self.assertTrue(self.test.check_performance())

        self.test.sanity_patterns = None
        self.test.perf_patterns = None
        self.assertTrue(self.test.check_sanity())
        self.assertTrue(self.test.check_performance())

    def test_file_not_found(self):
        self.test.stagedir = self.test.prefix
        self.test.perf_patterns = {
            'foobar': {
                'performance1 = (?P<value1>\S+)': [
                    ('value1', float, standard_threshold)
                ],
                'performance2 = (?P<value2>\S+)': [
                    ('value2', float, standard_threshold)
                ],
                'performance3 = (?P<value3>\S+)': [
                    ('value3', float, standard_threshold)
                ],
            }
        }

        self.assertFalse(self.test.check_performance())
