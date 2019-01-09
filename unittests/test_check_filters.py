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
        #TODO: add check
        self.checks = [
            self.create_check({'name': 'check1',
                'tags': {'a', 'b', 'c', 'd'},
                'valid_prog_environs': ['env1', 'env2'],
                'valid_systems': ['testsys:gpu', 'testsys2:mc'],
                'num_gpus_per_node': 1}),
            self.create_check({'name': 'check2',
                'tags': {'x', 'y', 'z'},
                'valid_prog_environs': ['env3'],
                'valid_systems': ['testsys:mc', 'testsys2:mc'],
                'num_gpus_per_node': 0})]

    def test_have_name(self):
        self.assertEqual(1, sn.count(filter(filters.have_name('check1'), self.checks)))
        self.assertEqual(1, sn.count(filter(filters.have_name('check2'), self.checks)))
        self.assertEqual(0, sn.count(filter(filters.have_name('check3'), self.checks)))
        self.assertEqual(0, sn.count(filter(filters.have_name('check4'), self.checks)))

    def test_have_not_name(self):
        self.assertEqual(1, sn.count(filter(filters.have_not_name('check1'), self.checks)))
        self.assertEqual(1, sn.count(filter(filters.have_not_name('check2'), self.checks)))
        self.assertEqual(2, sn.count(filter(filters.have_not_name('check3'), self.checks)))
        self.assertEqual(2, sn.count(filter(filters.have_not_name('check4'), self.checks)))
              
    def test_have_tags(self):
        self.assertEqual(1, sn.count(filter(filters.have_tag(['a', 'c']), self.checks)))
        self.assertEqual(0, sn.count(filter(filters.have_tag(['a', 'b', 'c', 'd', 'e']), self.checks)))
        self.assertEqual(1, sn.count(filter(filters.have_tag(['x', 'y', 'z']), self.checks)))
        self.assertEqual(0, sn.count(filter(filters.have_tag(['d', 'y']), self.checks)))

    def test_have_prgenv(self):
        self.assertEqual(1, sn.count(filter(filters.have_prgenv(['env1', 'env2']), self.checks)))
        self.assertEqual(1, sn.count(filter(filters.have_prgenv(['env3']), self.checks)))
        self.assertEqual(0, sn.count(filter(filters.have_prgenv(['env4']), self.checks)))
        self.assertEqual(0, sn.count(filter(filters.have_prgenv(['env1', 'env3']), self.checks)))

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_partition(self):
        p = rt.runtime().system.partition('gpu')
        self.assertEqual(1, sn.count(filter(filters.have_partition([p]), self.checks)))
        p = rt.runtime().system.partition('login')
        self.assertEqual(0, sn.count(filter(filters.have_partition([p]), self.checks)))

    def test_have_gpu_only(self):
        self.assertEqual(1, sn.count(filter(filters.have_gpu_only(), self.checks)))

    def test_have_cpu_only(self):
        self.assertEqual(1, sn.count(filter(filters.have_cpu_only(), self.checks)))
