# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import pytest

import reframe.core.config as config
from reframe.core.exceptions import ConfigError
from reframe.core.systems import System


def test_load_config_fallback(monkeypatch):
    monkeypatch.setattr(config, '_find_config_file', lambda: None)
    site_config = config.load_config()
    assert site_config.filename == '<builtin>'


def test_load_config_python():
    config.load_config('reframe/core/settings.py')


def test_load_config_python_old_syntax():
    with pytest.raises(ConfigError):
        config.load_config('unittests/resources/settings_old_syntax.py')


def test_load_config_nouser(monkeypatch):
    import pwd

    # Monkeypatch to simulate a system with no username
    monkeypatch.setattr(pwd, 'getpwuid', lambda uid: None)
    monkeypatch.delenv('LOGNAME', raising=False)
    monkeypatch.delenv('USER', raising=False)
    monkeypatch.delenv('LNAME', raising=False)
    monkeypatch.delenv('USERNAME', raising=False)
    config.load_config()


def test_convert_old_config():
    converted = config.convert_old_config(
        'unittests/resources/settings_old_syntax.py'
    )
    site_config = config.load_config(converted)
    site_config.validate()
    assert len(site_config.get('systems')) == 3
    assert len(site_config.get('logging')) == 2

    site_config.select_subconfig('testsys')
    assert len(site_config.get('systems/0/partitions')) == 2
    assert len(site_config.get('modes')) == 1
    assert len(site_config['environments']) == 6


def test_load_config_python_invalid(tmp_path):
    pyfile = tmp_path / 'settings.py'
    pyfile.write_text('x = 1\n')
    with pytest.raises(ConfigError,
                       match=r'not a valid Python configuration file'):
        config.load_config(pyfile)


def test_load_config_json(tmp_path):
    import reframe.core.settings as settings

    json_file = tmp_path / 'settings.json'
    json_file.write_text(json.dumps(settings.site_configuration, indent=4))
    site_config = config.load_config(json_file)
    assert site_config.filename == json_file


def test_load_config_json_invalid_syntax(tmp_path):
    json_file = tmp_path / 'settings.json'
    json_file.write_text('foo')
    with pytest.raises(ConfigError, match=r'invalid JSON syntax'):
        config.load_config(json_file)


def test_load_config_unknown_file(tmp_path):
    with pytest.raises(OSError):
        config.load_config(tmp_path / 'foo.json')


def test_load_config_import_error():
    # If the configuration file is relative to ReFrame and ImportError is
    # raised, which should be wrapped inside ConfigError
    with pytest.raises(ConfigError,
                       match=r'could not load Python configuration file'):
        config.load_config('reframe/core/foo.py')


def test_load_config_unknown_filetype(tmp_path):
    import reframe.core.settings as settings

    json_file = tmp_path / 'foo'
    json_file.write_text(json.dumps(settings.site_configuration, indent=4))
    with pytest.raises(ConfigError, match=r'unknown configuration file type'):
        config.load_config(json_file)


def test_validate_fallback_config():
    site_config = config.load_config('reframe/core/settings.py')
    site_config.validate()


def test_validate_unittest_config():
    site_config = config.load_config('unittests/resources/settings.py')
    site_config.validate()


def test_validate_config_invalid_syntax():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'][0]['name'] = 123
    with pytest.raises(ConfigError,
                       match=r'could not validate configuration file'):
        site_config.validate()


def test_validate_config_duplicate_systems():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'].append(site_config['systems'][0])
    with pytest.raises(ConfigError,
                       match=r"system 'generic' already defined"):
        site_config.validate()


def test_validate_config_duplicate_partitions():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'][0]['partitions'].append(
        site_config['systems'][0]['partitions'][0]
    )
    with pytest.raises(ConfigError,
                       match=r"partition 'default' already defined"):
        site_config.validate()


def test_select_subconfig_autodetect():
    site_config = config.load_config('reframe/core/settings.py')
    site_config.select_subconfig()
    assert site_config['systems'][0]['name'] == 'generic'


def test_select_subconfig_autodetect_failure():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'][0]['hostnames'] = ['$^']
    with pytest.raises(
            ConfigError,
            match=(r'could not find a configuration entry '
                   'for the current system')
    ):
        site_config.select_subconfig()


def test_select_subconfig_unknown_system():
    site_config = config.load_config('reframe/core/settings.py')
    with pytest.raises(
            ConfigError,
            match=(r'could not find a configuration entry '
                   'for the requested system')
    ):
        site_config.select_subconfig('foo')


def test_select_subconfig_unknown_partition():
    site_config = config.load_config('reframe/core/settings.py')
    with pytest.raises(
            ConfigError,
            match=(r'could not find a configuration entry '
                   'for the requested system/partition')
    ):
        site_config.select_subconfig('generic:foo')


def test_select_subconfig_no_logging():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['logging'][0]['target_systems'] = ['foo']
    with pytest.raises(ConfigError, match=r"section 'logging' not defined"):
        site_config.select_subconfig()


def test_select_subconfig_no_environments():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['environments'][0]['target_systems'] = ['foo']
    with pytest.raises(ConfigError,
                       match=r"section 'environments' not defined"):
        site_config.select_subconfig()


def test_select_subconfig_undefined_environment():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'][0]['partitions'][0]['environs'] += ['foo', 'bar']
    with pytest.raises(
            ConfigError,
            match=r"environments ('foo', 'bar')|('bar', 'foo') are not defined"
    ):
        site_config.select_subconfig()


def test_select_subconfig_ignore_resolve_errors():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'][0]['partitions'][0]['environs'] += ['foo', 'bar']
    site_config.select_subconfig(ignore_resolve_errors=True)


def test_select_subconfig_ignore_no_section_errors():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['environments'][0]['target_systems'] = ['foo']
    site_config.select_subconfig(ignore_resolve_errors=True)


def test_select_subconfig():
    site_config = config.load_config('unittests/resources/settings.py')
    site_config.select_subconfig('testsys')
    assert len(site_config['systems']) == 1
    assert len(site_config['systems'][0]['partitions']) == 2
    assert len(site_config['modes']) == 1
    assert site_config.get('systems/0/name') == 'testsys'
    assert site_config.get('systems/0/descr') == 'Fake system for unit tests'
    assert site_config.get('systems/0/hostnames') == ['testsys']
    assert site_config.get('systems/0/prefix') == '.rfm_testing'
    assert (site_config.get('systems/0/resourcesdir') ==
            '.rfm_testing/resources')
    assert site_config.get('systems/0/modules') == [{'name': 'foo/1.0',
                                                     'collection': False,
                                                     'path': None}]
    assert site_config.get('systems/0/env_vars') == [['FOO_CMD', 'foobar']]
    assert site_config.get('systems/0/modules_system') == 'nomod'
    assert site_config.get('systems/0/outputdir') == ''
    assert site_config.get('systems/0/stagedir') == ''
    assert len(site_config.get('systems/0/partitions')) == 2
    assert site_config.get('systems/0/partitions/@gpu/max_jobs') == 10
    assert site_config.get('modes/0/name') == 'unittest'
    assert site_config.get('modes/@unittest/name') == 'unittest'
    assert len(site_config.get('logging/0/handlers')) == 2
    assert len(site_config.get('logging/0/handlers_perflog')) == 1
    assert site_config.get('logging/0/handlers/0/timestamp') is False
    assert site_config.get('logging/0/handlers/0/level') == 'debug'
    assert site_config.get('logging/0/handlers/1/level') == 'info'
    assert site_config.get('logging/0/handlers/2/level') is None

    site_config.select_subconfig('testsys:login')
    assert len(site_config.get('systems/0/partitions')) == 1
    assert site_config.get('systems/0/partitions/0/scheduler') == 'local'
    assert site_config.get('systems/0/partitions/0/launcher') == 'local'
    assert (site_config.get('systems/0/partitions/0/environs') ==
            ['PrgEnv-cray', 'PrgEnv-gnu']
            )
    assert site_config.get('systems/0/partitions/0/descr') == 'Login nodes'
    assert site_config.get('systems/0/partitions/0/resources') == []
    assert site_config.get('systems/0/partitions/0/access') == []
    assert site_config.get('systems/0/partitions/0/container_platforms') == [
        {'type': 'Sarus'},
        {'type': 'Docker', 'default': True},
        {'type': 'Singularity'}
    ]
    assert site_config.get('systems/0/partitions/0/modules') == []
    assert site_config.get('systems/0/partitions/0/env_vars') == []
    assert site_config.get('systems/0/partitions/0/max_jobs') == 8
    assert len(site_config['environments']) == 7
    assert site_config.get('environments/@PrgEnv-gnu/cc') == 'gcc'
    assert site_config.get('environments/0/cxx') == 'g++'
    assert site_config.get('environments/@PrgEnv-cray/cc') == 'cc'
    assert site_config.get('environments/1/cxx') == 'CC'
    assert (site_config.get('environments/@PrgEnv-cray/modules') ==
            [{'name': 'PrgEnv-cray', 'collection': False, 'path': None}]
            )
    assert site_config.get('environments/@PrgEnv-gnu/extras') == {'foo': 1,
                                                                  'bar': 'x'}
    assert site_config.get('environments/@PrgEnv-gnu/features') == ['cxx14']
    assert site_config.get('environments/@PrgEnv-cray/extras') == {}
    assert site_config.get('environments/@PrgEnv-cray/features') == ['cxx14',
                                                                     'mpi']

    assert len(site_config.get('general')) == 1
    assert site_config.get('general/0/check_search_path') == ['a:b']

    site_config.select_subconfig('testsys:gpu')
    assert site_config.get('systems/0/partitions/@gpu/scheduler') == 'slurm'
    assert site_config.get('systems/0/partitions/0/launcher') == 'srun'
    assert (site_config.get('systems/0/partitions/0/environs') ==
            ['PrgEnv-gnu', 'builtin'])
    assert site_config.get('systems/0/partitions/0/descr') == 'GPU partition'
    assert len(site_config.get('systems/0/partitions/0/resources')) == 2
    assert (site_config.get('systems/0/partitions/0/resources/@gpu/name') ==
            'gpu')
    assert site_config.get('systems/0/partitions/0/modules') == [
        {'name': 'foogpu', 'collection': False, 'path': '/foo'}
    ]
    assert (site_config.get('systems/0/partitions/0/env_vars') ==
            [['FOO_GPU', 'yes']])
    assert site_config.get('systems/0/partitions/0/max_jobs') == 10
    assert site_config.get('environments/@PrgEnv-gnu/cc') == 'cc'
    assert site_config.get('environments/0/cxx') == 'CC'
    assert site_config.get('general/0/check_search_path') == ['c:d']

    # Test default values for non-existent name-addressable objects
    # See https://github.com/reframe-hpc/reframe/issues/1339
    assert site_config.get('modes/@foo/options') == []
    assert site_config.get('modes/10/options') == []

    # Test inexistent options
    site_config.select_subconfig('testsys')
    assert site_config.get('systems/1/name') is None
    assert site_config.get('systems/0/partitions/gpu/name') is None
    assert site_config.get('environments/0/foo') is None

    # Test misplaced slashes or empty option
    assert site_config.get('systems/0/partitions/@gpu/launcher/') == 'srun'
    assert site_config.get('/systems/0/partitions') is None
    assert site_config.get('', 'foo') == 'foo'
    assert site_config.get(None, 'foo') == 'foo'


def test_select_subconfig_optional_section_absent():
    site_config = config.load_config('reframe/core/settings.py')
    site_config.select_subconfig()
    assert site_config.get('general/0/colorize') is True
    assert site_config.get('general/0/git_timeout') == 5
    assert site_config.get('general/verbose') == 0


def test_sticky_options():
    site_config = config.load_config('unittests/resources/settings.py')
    site_config.select_subconfig('testsys:login')
    site_config.add_sticky_option('environments/cc', 'clang')
    site_config.add_sticky_option('modes/options', ['foo'])
    assert site_config.is_sticky_option('modes/options')
    assert site_config.get('environments/@PrgEnv-gnu/cc') == 'clang'
    assert site_config.get('environments/@PrgEnv-cray/cc') == 'clang'
    assert site_config.get('environments/@PrgEnv-cray/cxx') == 'CC'
    assert site_config.get('modes/0/options') == ['foo']

    # Remove the sticky options
    site_config.remove_sticky_option('environments/cc')
    site_config.remove_sticky_option('modes/options')
    assert site_config.get('environments/@PrgEnv-gnu/cc') == 'gcc'
    assert site_config.get('environments/@PrgEnv-cray/cc') == 'cc'


def test_system_create():
    site_config = config.load_config('unittests/resources/settings.py')
    site_config.select_subconfig('testsys:gpu')
    system = System.create(site_config)
    assert system.name == 'testsys'
    assert system.descr == 'Fake system for unit tests'
    assert system.hostnames == ['testsys']
    assert system.modules_system.name == 'nomod'
    assert system.preload_environ.modules == ['foo/1.0']
    assert system.preload_environ.env_vars == {'FOO_CMD': 'foobar'}
    assert system.prefix == '.rfm_testing'
    assert system.stagedir == ''
    assert system.outputdir == ''
    assert system.resourcesdir == '.rfm_testing/resources'
    assert len(system.partitions) == 1

    partition = system.partitions[0]
    assert partition.name == 'gpu'
    assert partition.fullname == 'testsys:gpu'
    assert partition.descr == 'GPU partition'
    assert partition.scheduler.registered_name == 'slurm'
    assert partition.launcher_type.registered_name == 'srun'
    assert partition.access == []
    assert len(partition.container_environs) == 1
    assert partition.container_runtime == 'Sarus'
    assert partition.local_env.modules == ['foogpu']
    assert partition.local_env.modules_detailed == [{
        'name': 'foogpu', 'collection': False, 'path': '/foo'
    }]
    assert partition.local_env.env_vars == {'FOO_GPU': 'yes'}
    assert partition.max_jobs == 10
    assert partition.time_limit is None
    assert len(partition.environs) == 2
    assert partition.environment('PrgEnv-gnu').cc == 'cc'
    assert partition.environment('PrgEnv-gnu').cflags == []
    assert partition.environment('PrgEnv-gnu').extras == {'foo': 2, 'bar': 'y'}

    # Check resource instantiation
    resource_spec = partition.get_resource('gpu', num_gpus_per_node=16)
    assert resource_spec == ['--gres=gpu:16']

    resources_spec = partition.get_resource(
        'datawarp', capacity='100GB', stagein_src='/foo'
    )
    assert resources_spec == ['#DW jobdw capacity=100GB',
                              '#DW stage_in source=/foo']

    # Check processor info
    assert partition.processor.info is not None
    assert partition.processor.topology is not None
    assert partition.processor.arch == 'skylake'
    assert partition.processor.num_cpus == 8
    assert partition.processor.num_cpus_per_core == 2
    assert partition.processor.num_cpus_per_socket == 8
    assert partition.processor.num_sockets == 1
    assert partition.processor.num_cores == 4
    assert partition.processor.num_cores_per_socket == 4
    assert partition.processor.num_numa_nodes == 1
    assert partition.processor.num_cores_per_numa_node == 4

    # Select another subconfig and check that the default selection of
    # container runtime is done properly
    site_config.select_subconfig('testsys:login')
    system = System.create(site_config)
    assert system.partitions[0].container_runtime == 'Docker'


def test_variables(tmp_path):
    # Test that the old syntax using `variables` instead of `env_vars` still
    # works
    config_file = tmp_path / 'settings.py'
    with open(config_file, 'w') as fout:
        with open('unittests/resources/settings.py') as fin:
            fout.write(fin.read().replace('env_vars', 'variables'))

    site_config = config.load_config(config_file)
    site_config.validate()
    site_config.select_subconfig('testsys')
    assert site_config.get('systems/0/variables') == [['FOO_CMD', 'foobar']]
    assert site_config.get('systems/0/env_vars') == [['FOO_CMD', 'foobar']]

    site_config.select_subconfig('testsys:login')
    assert site_config.get('systems/0/partitions/0/variables') == []
    assert site_config.get('systems/0/partitions/0/env_vars') == []

    site_config.select_subconfig('testsys:gpu')
    assert (site_config.get('systems/0/partitions/0/variables') ==
            [['FOO_GPU', 'yes']])
    assert (site_config.get('systems/0/partitions/0/env_vars') ==
            [['FOO_GPU', 'yes']])

    # Test that system is created correctly
    system = System.create(site_config)
    assert system.preload_environ.env_vars == {'FOO_CMD': 'foobar'}
    assert system.partitions[0].local_env.env_vars == {'FOO_GPU': 'yes'}


def test_hostname_autodetection():
    # This exercises only the various execution paths

    # We set the autodetection method and we call `select_subconfig()` in
    # order to trigger the auto-detection
    site_config = config.load_config('unittests/resources/settings.py')
    for use_xthostname in (True, False):
        for use_fqdn in (True, False):
            site_config.set_autodetect_meth('hostname',
                                            use_fqdn=use_fqdn,
                                            use_xthostname=use_xthostname)
            site_config.select_subconfig()
