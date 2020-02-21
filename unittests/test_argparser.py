# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import unittest

from reframe.frontend.argparse import ArgumentParser


class TestArgumentParser(unittest.TestCase):
    def setUp(self):
        self.parser = ArgumentParser()
        self.foo_options = self.parser.add_argument_group('foo options')
        self.bar_options = self.parser.add_argument_group('bar options')
        self.foo_options.add_argument('-f', '--foo', dest='foo',
                                      action='store', default='FOO')
        self.foo_options.add_argument('--foolist', dest='foolist',
                                      action='append', default=[])
        self.foo_options.add_argument('--foobar', action='store_true')
        self.foo_options.add_argument('--unfoo', action='store_false')

        self.bar_options.add_argument('-b', '--bar', dest='bar',
                                      action='store', default='BAR')
        self.bar_options.add_argument('--barlist', dest='barlist',
                                      action='append', default=[])
        self.foo_options.add_argument('--barfoo', action='store_true')

    def test_arguments(self):
        with pytest.raises(ValueError):
            self.foo_options.add_argument(action='store', default='FOO')

        self.foo_options.add_argument('--foo-bar', action='store_true')
        self.foo_options.add_argument('--alist', action='append', default=[])
        options = self.parser.parse_args(['--foobar', '--foo-bar'])
        assert options.foobar
        assert options.foo_bar

    def test_parsing(self):
        options = self.parser.parse_args(
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
        options = self.parser.parse_args(
            '--bar beer --foolist any'.split(), options
        )
        assert 'name' == options.foo
        assert ['any'] == options.foolist
        assert not options.foobar
        assert not options.unfoo
        assert 'beer' == options.bar
        assert [] == options.barlist
        assert options.barfoo
