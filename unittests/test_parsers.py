import os
import shutil
import stat
import tempfile
import unittest
import warnings

from reframe.core.pipeline import RegressionTest
from reframe.core.exceptions import ReframeError, ReframeDeprecationWarning
from reframe.core.systems import System, SystemPartition
from reframe.frontend.loader import RegressionCheckLoader
from reframe.frontend.resources import ResourcesManager
from reframe.utility.functions import standard_threshold
from reframe.utility.parsers import *
from unittests.fixtures import get_setup_config


class StatefulParserTest(unittest.TestCase):
    def setUp(self):
        # Ignore deprecation warnings
        warnings.simplefilter('ignore', ReframeDeprecationWarning)

        self.system, self.partition, self.environ = get_setup_config()
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.resources = ResourcesManager(prefix=self.resourcesdir)
        self.test = RegressionTest('test_performance',
                                   'unittests/resources',
                                   resources=self.resources,
                                   system=self.system)

        # Mock up an already set up test
        self.test.setup(self.partition, self.environ)

        self.perf_file = tempfile.NamedTemporaryFile(mode='wt', delete=False)
        self.output_file = tempfile.NamedTemporaryFile(mode='wt', delete=False)
        self.test.perf_parser   = StatefulParser(callback=standard_threshold)
        self.test.sanity_parser = StatefulParser()
        self.test.reference = {
            'testsys': {
                'value': (2.0, -0.1, 0.1),
            },
        }

        self.test.perf_patterns = {
            self.perf_file.name: {
                'performance = (?P<value>\S+)': [
                    ('value', float, self.test.perf_parser.match)
                ],
                '\e': self.test.perf_parser.match_eof
            }
        }

        self.test.sanity_patterns = {
            self.output_file.name: {
                '(?P<result>result = success)': [
                    ('result', str, self.test.sanity_parser.match)
                ],
                '\e': self.test.sanity_parser.match_eof
            }
        }

    def tearDown(self):
        self.perf_file.close()
        self.output_file.close()
        os.remove(self.perf_file.name)
        os.remove(self.output_file.name)
        shutil.rmtree(self.resourcesdir)
        warnings.simplefilter('default', ReframeDeprecationWarning)

    def _add_parser_region(self):
        self.test.perf_patterns[self.perf_file.name].update({
            '(?P<switch>== ENABLE ==)': [
                ('switch', str, self.test.perf_parser.on)
            ],
            '(?P<switch>== DISABLE ==)': [
                ('switch', str, self.test.perf_parser.off)
            ]}
        )

        self.test.sanity_patterns[self.output_file.name].update({
            '(?P<switch>== ENABLE ==)': [
                ('switch', str, self.test.sanity_parser.on)
            ],
            '(?P<switch>== DISABLE ==)': [
                ('switch', str, self.test.sanity_parser.off)
            ]}
        )

        self.test.reference['*:switch'] = None

    def _write_marker_enable(self, file):
        file.write('== ENABLE ==\n')

    def _write_marker_disable(self, file):
        file.write('== DISABLE ==\n')

    def is_parser_clear(self, parser, **kwargs):
        return not parser.is_on


class TestStatefulParserPerformance(StatefulParserTest):
    def setUp(self):
        super().setUp()

    def _write_good_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 0.1\n')
            file.write('performance = 10.1\n')
            self._write_marker_enable(file)

        file.write('performance = 1.9\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 3.2\n')

        file.close()

    def _write_bad_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 1.9\n')
            file.write('performance = 2.1\n')
            self._write_marker_enable(file)

        file.write('performance = 0.1\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 2.0\n')

        file.close()

    def test_performance_success(self):
        self.test.perf_parser.on()
        self._write_good_performance(file=self.perf_file)
        self.assertTrue(self.test.check_performance())
        self.assertTrue(self.is_parser_clear(self.test.perf_parser))

    def test_performance_success_with_region(self):
        self._add_parser_region()
        self._write_good_performance(file=self.perf_file, with_region=True)
        self.assertTrue(self.test.check_performance())
        self.assertTrue(self.is_parser_clear(self.test.perf_parser))

    def test_performance_failure(self):
        self.test.perf_parser.on()
        self._write_bad_performance(file=self.perf_file)
        self.assertFalse(self.test.check_performance())
        self.assertTrue(self.is_parser_clear(self.test.perf_parser))

    def test_performance_failure_with_region(self):
        self._add_parser_region()
        self._write_bad_performance(file=self.perf_file, with_region=True)
        self.assertFalse(self.test.check_performance())
        self.assertTrue(self.is_parser_clear(self.test.perf_parser))

    def test_default_status(self):
        self._write_good_performance(file=self.perf_file)
        self.assertFalse(self.test.check_performance())
        self.assertTrue(self.is_parser_clear(self.test.perf_parser))

    def test_empty_file(self):
        self.perf_file.close()
        self.assertFalse(self.test.check_performance())
        self.assertTrue(self.is_parser_clear(self.test.perf_parser))


class TestStatefulParserSanity(StatefulParserTest):
    def setUp(self):
        super().setUp()

    def _write_good_sanity(self, file, with_region=False):
        if with_region:
            file.write('result = failure\n')
            file.write('result = failure\n')
            self._write_marker_enable(file)

        file.write('result = success\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('result = failure\n')

        file.close()

    def _write_bad_sanity(self, file, with_region=False):
        if with_region:
            file.write('result = success\n')
            file.write('result = success\n')
            self._write_marker_enable(file)

        file.write('result = failure\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('result = success\n')

        file.close()

    def test_sanity_success(self):
        self.test.sanity_parser.on()
        self._write_good_sanity(file=self.output_file)
        self.assertTrue(self.test.check_sanity())
        self.assertTrue(self.is_parser_clear(self.test.sanity_parser))

    def test_sanity_success_with_region(self):
        self._add_parser_region()
        self._write_good_sanity(file=self.output_file, with_region=True)
        self.assertTrue(self.test.check_sanity())
        self.assertTrue(self.is_parser_clear(self.test.sanity_parser))

    def test_sanity_failure(self):
        self.test.sanity_parser.on()
        self._write_bad_sanity(file=self.output_file)
        self.assertFalse(self.test.check_sanity())
        self.assertTrue(self.is_parser_clear(self.test.sanity_parser))

    def test_sanity_failure_with_region(self):
        self._add_parser_region()
        self._write_bad_sanity(file=self.output_file, with_region=True)
        self.assertFalse(self.test.check_sanity())
        self.assertTrue(self.is_parser_clear(self.test.sanity_parser))

    def test_default_status(self):
        self._write_good_sanity(file=self.output_file)
        self.assertFalse(self.test.check_sanity())
        self.assertTrue(self.is_parser_clear(self.test.sanity_parser))

    def test_empty_file(self):
        self.output_file.close()
        self.assertFalse(self.test.check_sanity())
        self.assertTrue(self.is_parser_clear(self.test.sanity_parser))


class TestSingleOccurrenceParser(TestStatefulParserPerformance):
    def setUp(self):
        super().setUp()
        self.test.perf_parser = SingleOccurrenceParser(
            callback=standard_threshold, nth_occurrence=3)

        self.test.perf_patterns = {
            self.perf_file.name: {
                'performance = (?P<value>\S+)': [
                    ('value', float, self.test.perf_parser.match)
                ],
                '\e': self.test.perf_parser.match_eof
            }
        }

    def _write_good_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 0.1\n')
            self._write_marker_enable(file)

        file.write('performance = 0.1\n')
        file.write('performance = 1.9\n')
        file.write('performance = 2.1\n')
        file.write('performance = 1.2\n')

        if with_region:
            self._write_marker_disable(file)

        file.close()

    def _write_bad_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 1.9\n')
            self._write_marker_enable(file)

        file.write('performance = 2.1\n')
        file.write('performance = 0.1\n')
        file.write('performance = 0.1\n')
        file.write('performance = 10.2\n')

        if with_region:
            self._write_marker_disable(file)

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.count != 0:
            return False

        return super().is_parser_clear(parser)


class TestCounterParser(TestStatefulParserSanity):
    def setUp(self):
        super().setUp()
        self.test.sanity_parser = CounterParser(num_matches=3)
        self.test.sanity_patterns = {
            self.output_file.name: {
                '(?P<nid>nid\d+)': [
                    ('nid', str, self.test.sanity_parser.match)
                ],
                '\e': self.test.sanity_parser.match_eof
            }
        }

    def _write_good_sanity(self, file, with_region=False):
        if with_region:
            self._write_marker_enable(file)

        file.write('nid12\n')
        file.write('nid1\n')
        file.write('foo\n')
        file.write('nid34\n')
        file.write('bar\n')
        file.write('nid65\n')
        file.write('nid001\n')

        if with_region:
            self._write_marker_disable(file)

        file.close()

    def _write_bad_sanity(self, file, with_region=False):
        if with_region:
            self._write_marker_enable(file)

        file.write('nid12\n')
        file.write('foo\n')
        file.write('bar\n')
        file.write('nid001\n')
        file.write('whatever\n')
        file.write('another line\n')

        if with_region:
            self._write_marker_disable(file)

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.count != 0 or parser.last_match is not None:
            return False

        return super().is_parser_clear(parser)


class TestCounterParserExactMatch(TestStatefulParserSanity):
    def setUp(self):
        super().setUp()
        self.test.sanity_parser = CounterParser(num_matches=3, exact=True)
        self.test.sanity_patterns = {
            self.output_file.name: {
                '(?P<nid>nid\d+)': [
                    ('nid', str, self.test.sanity_parser.match)
                ],
                '\e': self.test.sanity_parser.match_eof
            }
        }

    def _write_good_sanity(self, file, with_region=False):
        if with_region:
            file.write('nid123\n')
            self._write_marker_enable(file)

        file.write('nid12\n')
        file.write('nid1\n')
        file.write('foo\n')
        file.write('nid34\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('nid3213\n')

        file.close()

    def _write_bad_sanity(self, file, with_region=False):
        if with_region:
            self._write_marker_enable(file)

        file.write('nid12\n')
        file.write('foo\n')
        file.write('bar\n')
        file.write('nid001\n')
        file.write('nid54\n')
        file.write('nid1112\n')

        if with_region:
            self._write_marker_disable(file)

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.count != 0:
            return False

        if parser.last_match is not None:
            return False

        return True


class TestCounterParserLastOccurrence(TestStatefulParserPerformance):
    def setUp(self):
        super().setUp()
        self.test.perf_parser = CounterParser(callback=standard_threshold,
                                              num_matches=-1)
        self.test.perf_patterns = {
            self.perf_file.name: {
                'performance = (?P<value>\S+)': [
                    ('value', float, self.test.perf_parser.match)
                ],
                '\e': self.test.perf_parser.match_eof
            }
        }

    def _write_good_performance(self, file, with_region=False):
        if with_region:
            self._write_marker_enable(file)

        file.write('performance = 0.2\n')
        file.write('performance = 0.2\n')
        file.write('performance = 0.2\n')
        file.write('performance = 1.9\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 0.2\n')

        file.close()

    def _write_bad_performance(self, file, with_region=False):
        if with_region:
            self._write_marker_enable(file)

        file.write('performance = 2.1\n')
        file.write('performance = 2.1\n')
        file.write('performance = 2.1\n')
        file.write('performance = 1.0\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 2.1\n')

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.count != 0 or parser.last_match is not None:
            return False

        return super().is_parser_clear(parser)


class TestUniqueOccurrencesParser(TestStatefulParserSanity):
    def setUp(self):
        super().setUp()
        self.test.sanity_parser = UniqueOccurrencesParser(num_matches=3)
        self.test.sanity_patterns = {
            self.output_file.name: {
                '(?P<nid>nid\d+)': [
                    ('nid', str, self.test.sanity_parser.match)
                ],
                '\e': self.test.sanity_parser.match_eof
            }
        }

    def _write_good_sanity(self, file, with_region=False):
        if with_region:
            file.write('nid009\n')
            self._write_marker_enable(file)

        file.write('nid001\n')
        file.write('nid002\n')
        file.write('foo\n')
        file.write('nid003\n')
        file.write('nid003\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('nid004\n')

        file.close()

    def _write_bad_sanity(self, file, with_region=False):
        if with_region:
            file.write('nid001\n')
            self._write_marker_enable(file)

        file.write('nid002\n')
        file.write('foo\n')
        file.write('bar\n')
        file.write('nid002\n')
        file.write('nid003\n')
        file.write('nid003\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('nid004\n')

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        return not parser.matched


class TestMinParser(TestStatefulParserPerformance):
    def setUp(self):
        super().setUp()
        self.test.perf_parser = MinParser(callback=standard_threshold)
        self.test.perf_patterns = {
            self.perf_file.name: {
                'performance = (?P<value>\S+)': [
                    ('value', float, self.test.perf_parser.match)
                ],
                '\e': self.test.perf_parser.match_eof
            }
        }

    def _write_good_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 0.2\n')
            self._write_marker_enable(file)

        file.write('performance = 2.1\n')
        file.write('performance = 3.1\n')
        file.write('performance = 4.1\n')
        file.write('performance = 5.1\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 0.1\n')

        file.close()

    def _write_bad_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 2.1\n')
            self._write_marker_enable(file)

        file.write('performance = 3.1\n')
        file.write('performance = 4.1\n')
        file.write('performance = 5.1\n')
        file.write('performance = 6.0\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 1.9\n')

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.value is not None:
            return False

        if parser.reference is not None:
            return False

        return True


class TestMaxParser(TestStatefulParserPerformance):
    def setUp(self):
        super().setUp()
        self.test.perf_parser = MaxParser(callback=standard_threshold)
        self.test.perf_patterns = {
            self.perf_file.name: {
                'performance = (?P<value>\S+)': [
                    ('value', float, self.test.perf_parser.match)
                ],
                '\e': self.test.perf_parser.match_eof
            }
        }

    def _write_good_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 10.1\n')
            self._write_marker_enable(file)

        file.write('performance = 2.1\n')
        file.write('performance = 1.2\n')
        file.write('performance = 1.3\n')
        file.write('performance = 1.4\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 9.1\n')

        file.close()

    def _write_bad_performance(self, file, with_region=False):
        if with_region:
            file.write('performance = 2.1\n')
            self._write_marker_enable(file)

        file.write('performance = 1.0\n')
        file.write('performance = 1.1\n')
        file.write('performance = 1.2\n')
        file.write('performance = 1.3\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('performance = 1.9\n')

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.value is not None:
            return False

        if parser.reference is not None:
            return False

        return True


class TestSumParser(TestStatefulParserPerformance):
    def setUp(self):
        super().setUp()
        self.test.perf_parser = SumParser(
            callback=lambda v, r, **kwargs: v == 10
        )
        self.test.perf_patterns = {
            self.perf_file.name: {
                'val = (?P<value>\S+)': [
                    ('value', int, self.test.perf_parser.match)
                ],
                '\e': self.test.perf_parser.match_eof
            }
        }

    def _write_good_performance(self, file, with_region=False):
        if with_region:
            file.write('val = 1\n')
            self._write_marker_enable(file)

        file.write('val = 1\n')
        file.write('val = 2\n')
        file.write('val = 3\n')
        file.write('val = 4\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('val = 1\n')

        file.close()

    def _write_bad_performance(self, file, with_region=False):
        if with_region:
            file.write('val = 4\n')
            self._write_marker_enable(file)

        file.write('val = 1\n')
        file.write('val = 2\n')
        file.write('val = 3\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('val = 4\n')

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.value is not None:
            return False

        if parser.reference is not None:
            return False

        return True


class TestAverageParser(TestStatefulParserPerformance):
    def setUp(self):
        super().setUp()
        self.test.perf_parser = AverageParser(callback=standard_threshold)
        self.test.perf_patterns = {
            self.perf_file.name: {
                'val = (?P<value>\S+)': [
                    ('value', float, self.test.perf_parser.match)
                ],
                '\e': self.test.perf_parser.match_eof
            }
        }

    def _write_good_performance(self, file, with_region=False):
        if with_region:
            file.write('val = 100\n')
            self._write_marker_enable(file)

        file.write('val = 1.9\n')
        file.write('val = 2.1\n')
        file.write('val = 1.9\n')
        file.write('val = 2.1\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('val = 100\n')

        file.close()

    def _write_bad_performance(self, file, with_region=False):
        if with_region:
            file.write('val = -100\n')
            self._write_marker_enable(file)

        file.write('val = 1.9\n')
        file.write('val = 2.1\n')
        file.write('val = 100\n')

        if with_region:
            self._write_marker_disable(file)
            file.write('val = -100\n')

        file.close()

    def is_parser_clear(self, parser, **kwargs):
        if parser.value is not None:
            return False

        if parser.reference is not None:
            return False

        if parser.count != 0:
            return False

        return True


class DeprecationWarningParserTest(unittest.TestCase):
    def assert_deprecation_warning(self, parser_cls, *args, **kwargs):
        with self.assertWarns(ReframeDeprecationWarning):
            fixture = parser_cls(*args, **kwargs)

    def test_statefulparser_deprecation(self):
        self.assert_deprecation_warning(StatefulParser)

    def test_counterparser_deprecation(self):
        self.assert_deprecation_warning(CounterParser, num_matches=1)

    def test_uniqueoccurencesparser_deprecation(self):
        self.assert_deprecation_warning(UniqueOccurrencesParser, num_matches=1)

    def test_singleoccurenceparser_deprecation(self):
        self.assert_deprecation_warning(SingleOccurrenceParser,
                                        nth_occurrence=1)

    def test_minparser_deprecation(self):
        self.assert_deprecation_warning(MinParser)

    def test_maxparser_deprecation(self):
        self.assert_deprecation_warning(MaxParser)

    def test_sumparser_deprecation(self):
        self.assert_deprecation_warning(SumParser)

    def test_averageparser_deprecation(self):
        self.assert_deprecation_warning(AverageParser)
