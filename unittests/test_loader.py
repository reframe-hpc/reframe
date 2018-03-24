import copy
import os
import unittest

from reframe.core.exceptions import ConfigError, NameConflictError
from reframe.core.systems import System
from reframe.frontend.loader import RegressionCheckLoader, SiteConfiguration
from reframe.frontend.resources import ResourcesManager
from unittests.fixtures import TEST_SITE_CONFIG


class TestSiteConfigurationFromDict(unittest.TestCase):
    def setUp(self):
        self.config = SiteConfiguration()
        self.site_config = copy.deepcopy(TEST_SITE_CONFIG)

    def test_load_success(self):
        self.config.load_from_dict(self.site_config)
        self.assertEqual(1, len(self.config.systems))

        system = self.config.systems['testsys']
        self.assertEqual(2, len(system.partitions))
        self.assertNotEqual(None, system.partition('login'))
        self.assertNotEqual(None, system.partition('gpu'))
        self.assertEqual('testsys:login', system.partition('login').fullname)
        self.assertEqual('testsys:gpu', system.partition('gpu').fullname)

        self.assertEqual(3, len(system.partition('login').environs))
        self.assertEqual(2, len(system.partition('gpu').environs))

        # Check that PrgEnv-gnu on login partition is resolved to the special
        # version defined in the 'dom:login' section
        env_login = system.partition('login').environment('PrgEnv-gnu')
        self.assertEqual('gcc', env_login.cc)
        self.assertEqual('g++', env_login.cxx)
        self.assertEqual('gfortran', env_login.ftn)

        # Check that the PrgEnv-gnu of the gpu partition is resolved to the
        # default one
        part = system.partition('gpu')
        env_gpu = part.environment('PrgEnv-gnu')
        self.assertEqual('cc', env_gpu.cc)
        self.assertEqual('CC', env_gpu.cxx)
        self.assertEqual('ftn', env_gpu.ftn)

        # Check resource instantiation
        self.assertEqual(['--gres=gpu:16'],
                         part.get_resource('gpu', num_gpus_per_node=16))
        self.assertEqual(['#DW jobdw capacity=100GB',
                          '#DW stage_in source=/foo'],
                         part.get_resource('datawarp',
                                           capacity='100GB',
                                           stagein_src='/foo'))

    def test_load_failure_empty_dict(self):
        site_config = {}
        self.assertRaises(ValueError,
                          self.config.load_from_dict, site_config)

    def test_load_failure_no_environments(self):
        site_config = {'systems': {}}
        self.assertRaises(ValueError,
                          self.config.load_from_dict, site_config)

    def test_load_failure_no_systems(self):
        site_config = {'environments': {}}
        self.assertRaises(ValueError,
                          self.config.load_from_dict, site_config)

    def test_load_failure_environments_no_scoped_dict(self):
        self.site_config['environments'] = {
            'testsys': 'PrgEnv-gnu'
        }
        self.assertRaises(TypeError,
                          self.config.load_from_dict, self.site_config)

    def test_load_failure_partitions_nodict(self):
        self.site_config['systems']['testsys']['partitions'] = ['gpu']
        self.assertRaises(ConfigError,
                          self.config.load_from_dict, self.site_config)

    def test_load_failure_systems_nodict(self):
        self.site_config['systems']['testsys'] = ['gpu']
        self.assertRaises(TypeError,
                          self.config.load_from_dict, self.site_config)

    def test_load_failure_partitions_nodict(self):
        self.site_config['systems']['testsys']['partitions']['login'] = 'foo'
        self.assertRaises(TypeError,
                          self.config.load_from_dict, self.site_config)

    def test_load_failure_partconfig_nodict(self):
        self.site_config['systems']['testsys']['partitions']['login'] = 'foo'
        self.assertRaises(TypeError,
                          self.config.load_from_dict, self.site_config)

    def test_load_failure_unresolved_environment(self):
        self.site_config['environments'] = {
            '*': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                }
            }
        }
        self.assertRaises(ConfigError,
                          self.config.load_from_dict, self.site_config)

    def test_load_failure_envconfig_nodict(self):
        self.site_config['environments']['*']['PrgEnv-gnu'] = 'foo'
        self.assertRaises(TypeError,
                          self.config.load_from_dict, self.site_config)

    def test_load_failure_envconfig_notype(self):
        self.site_config['environments'] = {
            '*': {
                'PrgEnv-gnu': {
                    'modules': ['PrgEnv-gnu'],
                }
            }
        }
        self.assertRaises(ConfigError,
                          self.config.load_from_dict, self.site_config)


class TestRegressionCheckLoader(unittest.TestCase):
    def setUp(self):
        self.loader = RegressionCheckLoader(['.'], ignore_conflicts=True)
        self.loader_with_path = RegressionCheckLoader(
            ['unittests/resources', 'unittests/foobar'],
            ignore_conflicts=True)
        self.loader_with_prefix = RegressionCheckLoader(
            load_path=['badchecks'],
            prefix=os.path.abspath('unittests/resources'))

        self.system = System('foo')
        self.resources = ResourcesManager()

    def test_load_file_relative(self):
        checks = self.loader.load_from_file(
            'unittests/resources/emptycheck.py',
            system=self.system, resources=self.resources
        )
        self.assertEqual(1, len(checks))
        self.assertEqual(checks[0].name, 'emptycheck')

    def test_load_file_absolute(self):
        checks = self.loader.load_from_file(
            os.path.abspath('unittests/resources/emptycheck.py'),
            system=self.system, resources=self.resources
        )
        self.assertEqual(1, len(checks))
        self.assertEqual(checks[0].name, 'emptycheck')

    def test_load_recursive(self):
        checks = self.loader.load_from_dir(
            'unittests/resources', recurse=True,
            system=self.system, resources=self.resources
        )
        self.assertEqual(12, len(checks))

    def test_load_all(self):
        checks = self.loader_with_path.load_all(system=self.system,
                                                resources=self.resources)
        self.assertEqual(11, len(checks))

    def test_load_all_with_prefix(self):
        checks = self.loader_with_prefix.load_all(system=self.system,
                                                  resources=self.resources)
        self.assertEqual(1, len(checks))

    def test_conflicted_checks(self):
        self.loader_with_path._ignore_conflicts = False
        self.assertRaises(NameConflictError, self.loader_with_path.load_all,
                          system=self.system, resources=self.resources)

    def test_load_error(self):
        self.assertRaises(OSError, self.loader.load_from_file,
                          'unittests/resources/foo.py')
