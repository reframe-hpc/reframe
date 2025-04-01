# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.runtime as rt
import unittests.utility as test_util
from reframe.frontend.argparse import ArgumentParser, CONST_DEFAULT


@pytest.fixture
def default_exec_ctx(make_exec_ctx_g):
    yield from make_exec_ctx_g(test_util.BUILTIN_CONFIG_FILE)


@pytest.fixture
def argparser(make_exec_ctx):
    make_exec_ctx()
    return ArgumentParser()


@pytest.fixture
def foo_options(argparser):
    opt_group = argparser.add_argument_group('foo options')
    opt_group.add_argument('-f', '--foo', dest='foo',
                           action='store', default='FOO')
    opt_group.add_argument('--foolist', dest='foolist',
                           action='append', default=[])
    opt_group.add_argument('--foobar', action='store_true')
    opt_group.add_argument('--unfoo', action='store_false')
    opt_group.add_argument('--barfoo', action='store_true')
    return opt_group


@pytest.fixture
def bar_options(argparser):
    opt_group = argparser.add_argument_group('bar options')
    opt_group.add_argument('-b', '--bar', dest='bar',
                           action='store', default='BAR')
    opt_group.add_argument('--barlist', dest='barlist',
                           action='append', default=[])
    return opt_group


def test_arguments(argparser, foo_options):
    with pytest.raises(ValueError):
        foo_options.add_argument(action='store', default='FOO')

    foo_options.add_argument('--foo-bar', action='store_true')
    foo_options.add_argument('--alist', action='append', default=[])
    assert foo_options.has_known_options(['-f', '-l', '-b'])
    assert foo_options.has_known_options(['--foobar', '-l', '-b'])
    assert not foo_options.has_known_options(['-l', '-b'])

    options = argparser.parse_args(['--foobar', '--foo-bar'])
    assert options.foobar
    assert options.foo_bar


def test_parsing(argparser, foo_options, bar_options):
    options = argparser.parse_args(
        '--foo name --foolist gag --barfoo --unfoo'.split()
    )
    assert 'name' == options.foo
    assert ['gag'] == options.foolist
    assert options.barfoo
    assert not options.unfoo

    # Check the defaults now
    assert not options.foobar
    assert 'BAR' == options.bar
    assert [] == options.barlist

    # Reparse based on the already loaded options
    options = argparser.parse_args(
        '--bar beer --foolist any'.split(), options
    )
    assert 'name' == options.foo

    # 'append' options are extended
    assert ['gag', 'any'] == options.foolist
    assert not options.foobar
    assert not options.unfoo
    assert 'beer' == options.bar
    assert [] == options.barlist
    assert options.barfoo


@pytest.fixture
def extended_parser():
    parser = ArgumentParser()
    foo_options = parser.add_argument_group('Foo options')
    bar_options = parser.add_argument_group('Bar options')
    parser.add_argument(
        '-R', '--recursive', action='store_true',
        envvar='RFM_RECURSIVE', configvar='general/check_search_recursive'
    )
    parser.add_argument(
        '--non-default-craype', action='store_true',
        envvar='RFM_NON_DEFAULT_CRAYPE', configvar='general/non_default_craype'
    )
    parser.add_argument(
        '--prefix', action='store', configvar='systems/prefix'
    )
    parser.add_argument('--version', action='version', version='1.0')
    parser.add_argument(
        dest='keep_stage_files', action='store_true',
        envvar='RFM_KEEP_STAGE_FILES', configvar='general/keep_stage_files'
    )
    parser.add_argument(
        '--git-timeout', envvar='RFM_GIT_TIMEOUT', action='store',
        configvar='general/git_timeout', type=float
    )

    # Option that is only associated with an environment variable
    parser.add_argument(
        dest='env_option',
        envvar='RFM_ENV_OPT',
        action='store',
        default='bar'
    )
    foo_options.add_argument(
        '--timestamp', action='store', nargs='?', const=CONST_DEFAULT,
        envvar='RFM_TIMESTAMP_DIRS', configvar='general/timestamp_dirs'
    )
    foo_options.add_argument(
        '-C', '--config-file', action='store', envvar='RFM_CONFIG_FILES'
    )
    foo_options.add_argument(
        '--check-path', action='append', envvar='RFM_CHECK_SEARCH_PATH :'
    )
    foo_options.add_argument(
        '--stagedir', action='store', configvar='systems/stagedir',
        default='/foo'
    )
    bar_options.add_argument(
        '--module', action='append', envvar='RFM_MODULES_PRELOAD'
    )
    bar_options.add_argument(
        '--nocolor', action='store_false', dest='colorize',
        envvar='RFM_COLORIZE', configvar='general/colorize'
    )
    return parser


def test_option_precedence(default_exec_ctx, extended_parser):
    with rt.temp_environment(env_vars={
            'RFM_NON_DEFAULT_CRAYPE': 'yes',
            'RFM_MODULES_PRELOAD': 'a,b,c',
            'RFM_CHECK_SEARCH_PATH': 'x:y:z'

    }):
        options = extended_parser.parse_args(['--nocolor', '--timestamp'])
        assert options.recursive is None
        assert options.timestamp is CONST_DEFAULT
        assert options.non_default_craype is True
        assert options.config_file is None
        assert options.prefix is None
        assert options.stagedir == '/foo'
        assert options.module == ['a', 'b', 'c']
        assert options.check_path == ['x', 'y', 'z']
        assert options.colorize is False


def test_option_with_config(default_exec_ctx, extended_parser, tmp_path):
    with rt.temp_environment(env_vars={
            'RFM_TIMESTAMP_DIRS': r'%F',
            'RFM_NON_DEFAULT_CRAYPE': 'yes',
            'RFM_MODULES_PRELOAD': 'a,b,c',
            'RFM_KEEP_STAGE_FILES': 'no',
            'RFM_GIT_TIMEOUT': '0.3'
    }):
        site_config = rt.runtime().site_config
        options = extended_parser.parse_args(['--nocolor', '--timestamp'])
        options.update_config(site_config)
        assert site_config.get('general/0/check_search_recursive') is False
        assert site_config.get('general/0/timestamp_dirs') == r'%F'
        assert site_config.get('general/0/non_default_craype') is True
        assert site_config.get('systems/0/prefix') == str(tmp_path)
        assert site_config.get('general/0/colorize') is False
        assert site_config.get('general/0/keep_stage_files') is False
        assert site_config.get('general/0/git_timeout') == 0.3

        # Defaults specified in parser override those in configuration file
        assert site_config.get('systems/0/stagedir') == '/foo'


def test_option_envvar_conversion_error(default_exec_ctx, extended_parser):
    with rt.temp_environment(env_vars={
            'RFM_NON_DEFAULT_CRAYPE': 'foo',
            'RFM_GIT_TIMEOUT': 'non-float'
    }):
        site_config = rt.runtime().site_config
        options = extended_parser.parse_args(['--nocolor'])
        errors = options.update_config(site_config)
        assert len(errors) == 2


def test_envvar_option(default_exec_ctx, extended_parser):
    with rt.temp_environment(env_vars={'RFM_ENV_OPT': 'BAR'}):
        options = extended_parser.parse_args([])
        assert options.env_option == 'BAR'


def test_envvar_option_default_val(default_exec_ctx, extended_parser):
    options = extended_parser.parse_args([])
    assert options.env_option == 'bar'


def test_suppress_required(argparser):
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument('--foo', action='store_true')
    group.add_argument('--bar', action='store_true')
    argparser.parse_args([], suppress_required=True)
