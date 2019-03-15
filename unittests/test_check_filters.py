import unittest

import reframe.core.runtime as rt
import reframe.frontend.check_filters as filters
import reframe.utility.sanity as sn
import unittests.fixtures as fixtures
from reframe.core.pipeline import RegressionTest


class TestCheckFilters(unittest.TestCase):
    def create_check(self, attrs):
        ret = RegressionTest()
        for k, v in attrs.items():
            setattr(ret, k, v)

        return ret

    def setUp(self):
        self.checks = [
            self.create_check({
                'name': 'check1',
                'tags': {'a', 'b', 'c', 'd'},
                'valid_prog_environs': ['env1', 'env2'],
                'valid_systems': ['testsys:gpu', 'testsys2:mc'],
                'num_gpus_per_node': 1}),
            self.create_check({
                'name': 'check2',
                'tags': {'x', 'y', 'z'},
                'valid_prog_environs': ['env3'],
                'valid_systems': ['testsys:mc', 'testsys2:mc'],
                'num_gpus_per_node': 0}),
            self.create_check({
                'name': 'check3',
                'tags': {'a', 'z'},
                'valid_prog_environs': ['env3', 'env4'],
                'valid_systems': ['testsys:gpu'],
                'num_gpus_per_node': 1})
        ]

    def count_checks(self, filter_fn):
        return sn.count(filter(filter_fn, self.checks))

    def test_have_name(self):
        self.assertEqual(1, self.count_checks(filters.have_name('check1')))
        self.assertEqual(3, self.count_checks(filters.have_name('check')))
        self.assertEqual(2, self.count_checks(filters.have_name(r'\S*1|\S*3')))
        self.assertEqual(0, self.count_checks(filters.have_name('Check')))
        self.assertEqual(3, self.count_checks(filters.have_name('(?i)Check')))
        self.assertEqual(
            2, self.count_checks(filters.have_name('check1|(?i)CHECK2'))
        )

    def test_have_not_name(self):
        self.assertEqual(2, self.count_checks(filters.have_not_name('check1')))
        self.assertEqual(
            1, self.count_checks(filters.have_not_name('check1|check3'))
        )
        self.assertEqual(
            0, self.count_checks(filters.have_not_name('check1|check2|check3'))
        )
        self.assertEqual(3, self.count_checks(filters.have_not_name('Check1')))
        self.assertEqual(
            2, self.count_checks(filters.have_not_name('(?i)Check1'))
        )

    def test_have_tags(self):
        self.assertEqual(2, self.count_checks(filters.have_tag('a|c')))
        self.assertEqual(0, self.count_checks(filters.have_tag('p|q')))
        self.assertEqual(2, self.count_checks(filters.have_tag('z')))

    def test_have_prgenv(self):
        self.assertEqual(
            1, self.count_checks(filters.have_prgenv('env1|env2'))
        )
        self.assertEqual(2, self.count_checks(filters.have_prgenv('env3')))
        self.assertEqual(1, self.count_checks(filters.have_prgenv('env4')))
        self.assertEqual(
            3, self.count_checks(filters.have_prgenv('env1|env3'))
        )

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_partition(self):
        p = rt.runtime().system.partition('gpu')
        self.assertEqual(2, self.count_checks(filters.have_partition([p])))
        p = rt.runtime().system.partition('login')
        self.assertEqual(0, self.count_checks(filters.have_partition([p])))

    def test_have_gpu_only(self):
        self.assertEqual(2, self.count_checks(filters.have_gpu_only()))

    def test_have_cpu_only(self):
        self.assertEqual(1, self.count_checks(filters.have_cpu_only()))
