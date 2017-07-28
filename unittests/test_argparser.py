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
        self.assertRaises(ValueError, self.foo_options.add_argument,
                          action='store', default='FOO')
        self.foo_options.add_argument('--foo-bar', action='store_true')
        self.foo_options.add_argument('--alist', action='append', default=[])
        options = self.parser.parse_args([ '--foobar', '--foo-bar'])
        self.assertTrue(options.foobar)
        self.assertTrue(options.foo_bar)


    def test_parsing(self):
        options = self.parser.parse_args(
            '--foo name --foolist gag --barfoo --unfoo'.split()
        )
        self.assertEqual('name', options.foo)
        self.assertEqual(['gag'], options.foolist)
        self.assertTrue(options.barfoo)
        self.assertFalse(options.unfoo)

        # Check the defaults now
        self.assertFalse(options.foobar)
        self.assertEqual('BAR', options.bar)
        self.assertEqual([], options.barlist)

        # Reparse based on the already loaded options
        options = self.parser.parse_args(
            '--bar beer --foolist any'.split(), options
        )
        self.assertEqual('name', options.foo)
        self.assertEqual(['any'], options.foolist)
        self.assertFalse(options.foobar)
        self.assertFalse(options.unfoo)
        self.assertEqual('beer', options.bar)
        self.assertEqual([], options.barlist)
        self.assertTrue(options.barfoo)
