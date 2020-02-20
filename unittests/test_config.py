# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import unittest

import reframe.core.config as config
import unittests.fixtures as fixtures
from reframe.core.exceptions import ConfigError
import pytest


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
        assert len(self.site_config.systems) == 3

        system = self.site_config.systems['testsys']
        assert len(system.partitions) == 2
        assert system.prefix == '.rfm_testing'
        assert system.resourcesdir == '.rfm_testing/resources'
        assert system.perflogdir == '.rfm_testing/perflogs'
        assert system.preload_environ.modules == ['foo/1.0']
        assert system.preload_environ.variables == {'FOO_CMD': 'foobar'}

        part_login = self.get_partition(system, 'login')
        part_gpu = self.get_partition(system, 'gpu')
        assert part_login is not None
        assert part_gpu is not None
        assert part_login.fullname == 'testsys:login'
        assert part_gpu.fullname == 'testsys:gpu'
        assert len(part_login.environs) == 3
        assert len(part_gpu.environs) == 2

        # Check local partition environment
        assert part_gpu.local_env.modules == ['foogpu']
        assert part_gpu.local_env.variables == {'FOO_GPU': 'yes'}

        # Check that PrgEnv-gnu on login partition is resolved to the special
        # version defined in the 'dom:login' section
        env_login = part_login.environment('PrgEnv-gnu')
        assert env_login.cc == 'gcc'
        assert env_login.cxx == 'g++'
        assert env_login.ftn == 'gfortran'

        # Check that the PrgEnv-gnu of the gpu partition is resolved to the
        # default one
        env_gpu = part_gpu.environment('PrgEnv-gnu')
        assert env_gpu.cc == 'cc'
        assert env_gpu.cxx == 'CC'
        assert env_gpu.ftn == 'ftn'

        # Check resource instantiation
        resource_spec = part_gpu.get_resource('gpu', num_gpus_per_node=16)
        assert (resource_spec == ['--gres=gpu:16'])

        resources_spec = part_gpu.get_resource('datawarp',
                                               capacity='100GB',
                                               stagein_src='/foo')
        assert (resources_spec == ['#DW jobdw capacity=100GB',
                                   '#DW stage_in source=/foo'])

    def test_load_envconfig_with_unknown_args(self):
        self.dict_config['environments']['*']['builtin-gcc'] = {
            'foo': 'bar',
        }
        self.site_config.load_from_dict(self.dict_config)

    def test_load_failure_empty_dict(self):
        dict_config = {}
        with pytest.raises(ValueError):
            self.site_config.load_from_dict(dict_config)

    def test_load_failure_no_environments(self):
        dict_config = {'systems': {}}
        with pytest.raises(ValueError):
            self.site_config.load_from_dict(dict_config)

    def test_load_failure_no_systems(self):
        dict_config = {'environments': {}}
        with pytest.raises(ValueError):
            self.site_config.load_from_dict(dict_config)

    def test_load_failure_environments_no_scoped_dict(self):
        self.dict_config['environments'] = {
            'testsys': 'PrgEnv-gnu'
        }
        with pytest.raises(TypeError):
            self.site_config.load_from_dict(self.dict_config)

    def test_load_failure_partitions_nodict(self):
        self.dict_config['systems']['testsys']['partitions'] = ['gpu']
        with pytest.raises(ConfigError):
            self.site_config.load_from_dict(self.dict_config)

    def test_load_failure_systems_nodict(self):
        self.dict_config['systems']['testsys'] = ['gpu']
        with pytest.raises(TypeError):
            self.site_config.load_from_dict(self.dict_config)

    def test_load_failure_partitions_nodict(self):
        self.dict_config['systems']['testsys']['partitions']['login'] = 'foo'
        with pytest.raises(TypeError):
            self.site_config.load_from_dict(self.dict_config)

    def test_load_failure_partconfig_nodict(self):
        self.dict_config['systems']['testsys']['partitions']['login'] = 'foo'
        with pytest.raises(TypeError):
            self.site_config.load_from_dict(self.dict_config)

    def test_load_failure_unresolved_environment(self):
        self.dict_config['environments'] = {
            '*': {
                'PrgEnv-gnu': {
                    'modules': ['PrgEnv-gnu'],
                }
            }
        }
        with pytest.raises(ConfigError,
                           match='could not find a definition for'):
            self.site_config.load_from_dict(self.dict_config)

    def test_load_failure_envconfig_nodict(self):
        self.dict_config['environments']['*']['PrgEnv-gnu'] = 'foo'
        with pytest.raises(TypeError):
            self.site_config.load_from_dict(self.dict_config)


class TestConfigLoading(unittest.TestCase):
    def test_load_normal_config(self):
        config.load_settings_from_file('unittests/resources/settings.py')

    def test_load_unknown_file(self):
        with pytest.raises(ConfigError):
            config.load_settings_from_file('foo')

    def test_load_no_settings(self):
        with pytest.raises(ConfigError):
            config.load_settings_from_file('unittests')

    def test_load_invalid_settings(self):
        with pytest.raises(ConfigError):
            config.load_settings_from_file(
                'unittests/resources/invalid_settings.py')
