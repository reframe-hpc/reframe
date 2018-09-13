import copy
import unittest

import reframe.core.config as config
import unittests.fixtures as fixtures
from reframe.core.exceptions import ConfigError


class TestSiteConfigurationFromDict(unittest.TestCase):
    def setUp(self):
        self.site_config = config.SiteConfiguration()
        self.dict_config = copy.deepcopy(fixtures.TEST_SITE_CONFIG)

    def get_partition(self, system, name):
        for p in system.partitions:
            if p.name == name:
                return p

    def test_load_success(self):
        self.site_config.load_from_dict(self.dict_config)
        self.assertEqual(2, len(self.site_config.systems))

        system = self.site_config.systems['testsys']
        self.assertEqual(2, len(system.partitions))
        self.assertEqual('.rfm_testing', system.prefix)
        self.assertEqual('.rfm_testing/resources', system.resourcesdir)
        self.assertEqual('.rfm_testing/perflogs', system.perflogdir)

        part_login = self.get_partition(system, 'login')
        part_gpu = self.get_partition(system, 'gpu')
        self.assertIsNotNone(part_login)
        self.assertIsNotNone(part_gpu)
        self.assertEqual('testsys:login', part_login.fullname)
        self.assertEqual('testsys:gpu', part_gpu.fullname)
        self.assertEqual(3, len(part_login.environs))
        self.assertEqual(2, len(part_gpu.environs))

        # Check that PrgEnv-gnu on login partition is resolved to the special
        # version defined in the 'dom:login' section
        env_login = part_login.environment('PrgEnv-gnu')
        self.assertEqual('gcc', env_login.cc)
        self.assertEqual('g++', env_login.cxx)
        self.assertEqual('gfortran', env_login.ftn)

        # Check that the PrgEnv-gnu of the gpu partition is resolved to the
        # default one
        env_gpu = part_gpu.environment('PrgEnv-gnu')
        self.assertEqual('cc', env_gpu.cc)
        self.assertEqual('CC', env_gpu.cxx)
        self.assertEqual('ftn', env_gpu.ftn)

        # Check resource instantiation
        self.assertEqual(['--gres=gpu:16'],
                         part_gpu.get_resource('gpu', num_gpus_per_node=16))
        self.assertEqual(['#DW jobdw capacity=100GB',
                          '#DW stage_in source=/foo'],
                         part_gpu.get_resource('datawarp',
                                               capacity='100GB',
                                               stagein_src='/foo'))

    def test_load_failure_empty_dict(self):
        dict_config = {}
        self.assertRaises(ValueError,
                          self.site_config.load_from_dict, dict_config)

    def test_load_failure_no_environments(self):
        dict_config = {'systems': {}}
        self.assertRaises(ValueError,
                          self.site_config.load_from_dict, dict_config)

    def test_load_failure_no_systems(self):
        dict_config = {'environments': {}}
        self.assertRaises(ValueError,
                          self.site_config.load_from_dict, dict_config)

    def test_load_failure_environments_no_scoped_dict(self):
        self.dict_config['environments'] = {
            'testsys': 'PrgEnv-gnu'
        }
        self.assertRaises(TypeError,
                          self.site_config.load_from_dict, self.dict_config)

    def test_load_failure_partitions_nodict(self):
        self.dict_config['systems']['testsys']['partitions'] = ['gpu']
        self.assertRaises(ConfigError,
                          self.site_config.load_from_dict, self.dict_config)

    def test_load_failure_systems_nodict(self):
        self.dict_config['systems']['testsys'] = ['gpu']
        self.assertRaises(TypeError,
                          self.site_config.load_from_dict, self.dict_config)

    def test_load_failure_partitions_nodict(self):
        self.dict_config['systems']['testsys']['partitions']['login'] = 'foo'
        self.assertRaises(TypeError,
                          self.site_config.load_from_dict, self.dict_config)

    def test_load_failure_partconfig_nodict(self):
        self.dict_config['systems']['testsys']['partitions']['login'] = 'foo'
        self.assertRaises(TypeError,
                          self.site_config.load_from_dict, self.dict_config)

    def test_load_failure_unresolved_environment(self):
        self.dict_config['environments'] = {
            '*': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': ['PrgEnv-gnu'],
                }
            }
        }
        self.assertRaises(ConfigError,
                          self.site_config.load_from_dict, self.dict_config)

    def test_load_failure_envconfig_nodict(self):
        self.dict_config['environments']['*']['PrgEnv-gnu'] = 'foo'
        self.assertRaises(TypeError,
                          self.site_config.load_from_dict, self.dict_config)

    def test_load_failure_envconfig_notype(self):
        self.dict_config['environments'] = {
            '*': {
                'PrgEnv-gnu': {
                    'modules': ['PrgEnv-gnu'],
                }
            }
        }
        self.assertRaises(ConfigError,
                          self.site_config.load_from_dict, self.dict_config)


class TestConfigLoading(unittest.TestCase):
    def test_load_normal_config(self):
        config.load_settings_from_file('unittests/resources/settings.py')

    def test_load_unknown_file(self):
        self.assertRaises(ConfigError, config.load_settings_from_file, 'foo')

    def test_load_no_settings(self):
        self.assertRaises(ConfigError,
                          config.load_settings_from_file, 'unittests')

    def test_load_invalid_settings(self):
        self.assertRaises(ConfigError, config.load_settings_from_file,
                          'unittests/resources/invalid_settings.py')
