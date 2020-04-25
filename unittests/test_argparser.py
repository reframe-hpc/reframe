# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.runtime as rt
import unittests.fixtures as fixtures
from reframe.frontend.argparse import ArgumentParser


@pytest.fixture
def argparser():
    with rt.temp_runtime(fixtures.TEST_CONFIG_FILE):
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
    assert ['any'] == options.foolist
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
    foo_options.add_argument(
        '--timestamp', action='store',
        envvar='RFM_TIMESTAMP_DIRS', configvar='general/timestamp_dirs'
    )
    foo_options.add_argument(
        '-C', '--config-file', action='store', envvar='RFM_CONFIG_FILE'
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


def test_option_precedence(extended_parser):
    with rt.temp_environment(variables={
            'RFM_TIMESTAMP': '%F',
            'RFM_NON_DEFAULT_CRAYPE': 'yes',
            'RFM_MODULES_PRELOAD': 'a,b,c',
            'RFM_CHECK_SEARCH_PATH': 'x:y:z'

    }):
        options = extended_parser.parse_args(
            ['--timestamp=%FT%T', '--nocolor']
        )
        assert options.recursive is None
        assert options.timestamp == '%FT%T'
        assert options.non_default_craype is True
        assert options.config_file is None
        assert options.prefix is None
        assert options.stagedir == '/foo'
        assert options.module == ['a', 'b', 'c']
        assert options.check_path == ['x', 'y', 'z']
        assert options.colorize is False


def test_option_with_config(extended_parser):
    with rt.temp_environment(variables={
            'RFM_TIMESTAMP': '%F',
            'RFM_NON_DEFAULT_CRAYPE': 'yes',
            'RFM_MODULES_PRELOAD': 'a,b,c',
            'RFM_KEEP_STAGE_FILES': 'no'
    }):
        site_config = rt.runtime().site_config
        options = extended_parser.parse_args(
            ['--timestamp=%FT%T', '--nocolor']
        )
        options.update_config(site_config)
        assert site_config.get('general/0/check_search_recursive') is True
        assert site_config.get('general/0/timestamp_dirs') == '%FT%T'
        assert site_config.get('general/0/non_default_craype') is True
        assert site_config.get('systems/0/prefix') == '.'
        assert site_config.get('general/0/colorize') is False
        assert site_config.get('general/0/keep_stage_files') is False

        # Defaults specified in parser override those in configuration file
        assert site_config.get('systems/0/stagedir') == '/foo'


def test_option_envvar_conversion_error(extended_parser):
    with rt.temp_environment(variables={
            'RFM_NON_DEFAULT_CRAYPE': 'foo',
    }):
        site_config = rt.runtime().site_config
        options = extended_parser.parse_args(['--nocolor'])
        errors = options.update_config(site_config)
        assert len(errors) == 1
