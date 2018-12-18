import unittest

import reframe.core.runtime as rt
import reframe.frontend.check_filters as filters
import unittests.fixtures as fixtures
from reframe.core.systems import SystemPartition
from reframe.core.pipeline import RegressionTest
from reframe.utility.typecheck import Str

class TestCheckFilters(unittest.TestCase):
    def setUp(self):
        #TODO: use correct initialization
        #check 1
        self.check1 = RegressionTest()
        self.check1.name = 'check1'
        self.check1.tags = {'a', 'b', 'c', 'd'}
        self.check1.valid_prog_environs = ['env1', 'env2']
        self.check1.valid_systems = ['testsys:gpu', 'testsys2:mc']
        self.check1.num_gpus_per_node = 1
        #check 2
        self.check2 = RegressionTest()
        self.check2.name = 'check2'
        self.check2.tags = {'x', 'y', 'z'}
        self.check2.valid_prog_environs = ['env3']
        self.check2.valid_systems = ['testsys:mc', 'testsys2:mc']
        self.check2.num_gpus_per_node = 0

    def test_have_name(self):
        self.assertTrue(filters.have_name('check1')(self.check1))
        self.assertFalse(filters.have_name('check2')(self.check1))
        self.assertTrue(filters.have_name('check2')(self.check2))
        self.assertFalse(filters.have_name('check1')(self.check2))

    def test_have_not_name(self):
        self.assertTrue(filters.have_not_name('check2')(self.check1))
        self.assertFalse(filters.have_not_name('check1')(self.check1))
        self.assertTrue(filters.have_not_name('check1')(self.check2))
        self.assertFalse(filters.have_not_name('check2')(self.check2))
              
    def test_have_tags(self):
        self.assertTrue(filters.have_tag(['a', 'c'])(self.check1))
        self.assertFalse(filters.have_tag(['a', 'b', 'c', 'd', 'e'])(self.check1))
        self.assertFalse(filters.have_tag(['x', 'y', 'z'])(self.check1))
        self.assertFalse(filters.have_tag(['a', 'c'])(self.check2))
        self.assertFalse(filters.have_tag(['a', 'b', 'c', 'd', 'e'])(self.check2))
        self.assertTrue(filters.have_tag(['x', 'y', 'z'])(self.check2))

    def test_have_prgenv(self):
        self.assertTrue(filters.have_prgenv(['env1', 'env2'])(self.check1))
        self.assertFalse(filters.have_prgenv(['env3'])(self.check1))
        self.assertTrue(filters.have_prgenv(['env3'])(self.check2))
        self.assertFalse(filters.have_prgenv(['env1', 'env2'])(self.check2))

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_system(self):
        self.assertTrue(filters.have_system(rt.runtime().system.partitions)(self.check1))
        self.assertFalse(filters.have_system(rt.runtime().system.partitions)(self.check2))

    def test_have_gpu_only(self):
        self.assertTrue(filters.have_gpu_only()(self.check1))
        self.assertFalse(filters.have_gpu_only()(self.check2))

    def test_have_cpu_only(self):
        self.assertFalse(filters.have_cpu_only()(self.check1))
        self.assertTrue(filters.have_cpu_only()(self.check2))
