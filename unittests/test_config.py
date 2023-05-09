# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import pytest
import sys

import reframe.core.config as config
import reframe.utility as util
from reframe.core.exceptions import ConfigError
from reframe.core.systems import System


@pytest.fixture
def generate_partial_configs(tmp_path):
    part1 = tmp_path / 'settings-part1.py'
    part2 = tmp_path / 'settings-part2.py'
    part3 = tmp_path / 'settings-part3.py'
    mod = util.import_module_from_file(
        'unittests/resources/config/settings.py'
    )
    full_config = mod.site_configuration

    config_1 = {
        'systems': full_config['systems'][-2:],
        'environments': full_config['environments'][:3],
        'modes': full_config['modes'],
        'general': full_config['general'][0:1],
    }
    config_2 = {
        'systems': full_config['systems'][-3:-2],
        'environments': full_config['environments'][3:],
        'general': full_config['general'][1:3],
    }
    config_3 = {
        'systems': full_config['systems'][:-3],
        'logging': full_config['logging'],
        'general': full_config['general'][3:],
    }

    # We need to make sure that the custom hostname function is in one of the
    # files
    part1.write_text(f'def hostname(): return "testsys"\n'
                     f'site_configuration = {config_1!r}')
    part2.write_text(f'site_configuration = {config_2!r}')
    part3.write_text(f'site_configuration = {config_3!r}')
    return part1, part2, part3


@pytest.fixture(params=['full', 'parts'])
def site_config(request, generate_partial_configs):
    # `unittests/resources/config/settings.py` should be equivalent to loading
    # the `unittests/resources/config/settings-part*.py` files
    if request.param == 'full':
        return config.load_config('unittests/resources/config/settings.py')
    else:
        return config.load_config(*generate_partial_configs)


def test_load_config_python():
    site = config.load_config('reframe/core/settings.py')
    assert len(site.sources) == 2


def test_load_multiple_configs(generate_partial_configs):
    site = config.load_config(*generate_partial_configs)
    assert len(site.sources) == 4


def test_load_config_nouser(monkeypatch):
    import pwd

    # Monkeypatch to simulate a system with no username
    monkeypatch.setattr(pwd, 'getpwuid', lambda uid: None)
    monkeypatch.delenv('LOGNAME', raising=False)
    monkeypatch.delenv('USER', raising=False)
    monkeypatch.delenv('LNAME', raising=False)
    monkeypatch.delenv('USERNAME', raising=False)
    config.load_config()


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
    assert site_config.sources == ['<builtin>', json_file]


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
    site_config = config.load_config()
    site_config.validate()


def test_validate_unittest_config(site_config):
    site_config.validate()


def test_validate_config_invalid_syntax():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'][0]['name'] = 123
    with pytest.raises(ConfigError,
                       match=r'could not validate configuration file'):
        site_config.validate()


def test_select_subconfig_autodetect():
    site_config = config.load_config('reframe/core/settings.py')
    site_config.select_subconfig()
    assert site_config['systems'][0]['name'] == 'generic'


def test_select_subconfig_autodetect_failure():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['systems'][0]['hostnames'] = ['$^']
    site_config['systems'][1]['hostnames'] = ['$^']
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


def test_select_subconfig_no_environments():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['environments'][0]['target_systems'] = ['foo']
    site_config['environments'][1]['target_systems'] = ['foo']
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


def test_select_subconfig_empty_logging():
    site_config = config.load_config('reframe/core/settings.py')
    site_config['logging'][0] = {}
    with pytest.raises(ConfigError,
                       match=rf"'logging/handlers\$' are not defined"):
        site_config.select_subconfig()


def test_select_subconfig(site_config):
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
    assert len(site_config.get('logging/0/handlers$')) == 1
    assert len(site_config.get('logging/0/handlers')) == 1
    assert len(site_config.get('logging/0/handlers_perflog')) == 1
    assert site_config.get('logging/0/handlers/0/timestamp') is False
    assert site_config.get('logging/0/handlers/0/level') == 'debug'
    assert site_config.get('logging/0/handlers/1/level') is None

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
    assert site_config.get('environments/1/cxx') == 'g++'
    assert site_config.get('environments/@PrgEnv-cray/cc') == 'cc'
    assert site_config.get('environments/2/cxx') == 'CC'
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
    assert site_config.get('systems/0/partitions/0/sched_options') == {
        'use_nodes_option': True
    }
    assert site_config.get('systems/0/sched_options') == {
        'job_submit_timeout': 10
    }
    assert site_config.get('environments/@PrgEnv-gnu/cc') == 'cc'
    assert site_config.get('environments/1/cxx') == 'CC'
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


def test_sticky_options(site_config):
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


@pytest.fixture
def write_config(tmp_path):
    def _write_config(config):
        filename = (tmp_path / 'settings.json')
        filename.touch()
        with open(filename, 'w') as fp:
            json.dump(config, fp)

        return str(filename)

    return _write_config


def test_multi_config_combine_general_options(write_config):
    config_file = write_config({
        'general': [
            {
                'pipeline_timeout': 10,
                'target_systems': ['testsys:login']
            },
            {
                'colorize': False
            }
        ]
    })
    site_config = config.load_config('unittests/resources/config/settings.py',
                                     config_file)
    site_config.validate()
    site_config.select_subconfig('testsys:login')
    assert site_config.get('general/0/check_search_path') == ['a:b']
    assert site_config.get('general/0/pipeline_timeout') == 10
    assert site_config.get('general/0/colorize') == False


def test_multi_config_combine_logging_options(write_config):
    config_file = write_config({'logging': [{'level': 'debug'}]})
    site_config = config.load_config(config_file)
    site_config.validate()
    site_config.select_subconfig('generic')
    assert site_config.get('logging/0/level') == 'debug'
    assert len(site_config.get('logging/0/handlers')) == 1
    assert len(site_config.get('logging/0/handlers_perflog')) == 1


def test_system_create(site_config):
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
        with open('unittests/resources/config/settings.py') as fin:
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


def test_autodetect_meth_python(site_config, tmp_path):
    moduledir = tmp_path / 'mymod'
    moduledir.mkdir()
    with open(moduledir / '__init__.py', 'w') as fp:
        fp.write('def foo():\n\treturn "sys12"\n')

    site_config.set_autodetect_methods(['py::mymod.foo'])
    with util.temp_sys_path(str(tmp_path)):
        site_config.select_subconfig()

    assert site_config.get('systems/0/name') == 'sys0'

    # Uncache the module
    del sys.modules['mymod']


def test_autodetect_meth_python_inline(site_config):
    site_config.set_autodetect_methods(['py::hostname'])
    site_config.select_subconfig()
    assert site_config.get('systems/0/name') == 'testsys'


def test_autodetect_meth_python_errors(site_config):
    site_config.set_autodetect_methods(['py::mymod.foo'])
    with pytest.raises(ConfigError):
        site_config.select_subconfig()

    site_config.set_autodetect_methods(['py::foo'])
    with pytest.raises(ConfigError):
        site_config.select_subconfig()


def test_autodetect_meth_shell(site_config):
    site_config.set_autodetect_methods(['echo testsys'])
    site_config.select_subconfig()
    assert site_config.get('systems/0/name') == 'testsys'


def test_autodetect_meth_shell_errors(site_config):
    site_config.set_autodetect_methods(['xxxxxxxxx'])
    with pytest.raises(ConfigError):
        site_config.select_subconfig()


def test_autodetect_meth_multiple(site_config):
    site_config.set_autodetect_methods(['py::foo', 'echo testsys'])
    site_config.select_subconfig()
    assert site_config.get('systems/0/name') == 'testsys'


def test_hostname_autodetection(site_config):
    # This exercises only the various execution paths

    # We set the autodetection method and we call `select_subconfig()` in
    # order to trigger the auto-detection
    for use_xthostname in (True, False):
        for use_fqdn in (True, False):
            site_config.set_autodetect_methods(['py::socket.gethostname',
                                                'py::socket.getfqdn',
                                                'cat /etc/xthostname'])
            site_config.select_subconfig()
