import unittest

import reframe.core.runtime as rt
import reframe.frontend.check_filters as filters
import unittests.fixtures as fixtures
from reframe.core.systems import SystemPartition
from reframe.core.pipeline import RegressionTest
from reframe.utility.typecheck import Str

class TestCheckFilters(unittest.TestCase):
    def setUp(self):
        #TODO: use correct initialization and add checks
        self.check = RegressionTest()
        self.check.name = 'check1'
        self.check.tags = {'a', 'b', 'c', 'd'} 
        self.check.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        self.check.valid_systems = ['testsys:gpu', 'daint:mc']
        self.check.num_gpus_per_node = 1

    def test_have_name(self):
        self.assertTrue((filters.have_name('check1'))(self.check))
        self.assertFalse((filters.have_name('check2'))(self.check))

    def test_have_not_name(self):
        self.assertTrue((filters.have_not_name('check2'))(self.check))
        self.assertFalse((filters.have_not_name('check1'))(self.check))
              
    def test_have_tags(self):
        self.assertTrue((filters.have_tag(['a', 'c']))(self.check)) 
        self.assertFalse((filters.have_tag(['a', 'b', 'c', 'd', 'e']))(self.check)) 
        self.assertFalse((filters.have_tag(['x', 'y', 'z']))(self.check)) 

    def test_have_prgenv(self):
        self.assertTrue((filters.have_prgenv(['PrgEnv-cray', 'PrgEnv-gnu']))(self.check))
        self.assertFalse((filters.have_prgenv(['PrgEnv-pgi']))(self.check))

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_system(self):
        self.assertTrue((filters.have_system(rt.runtime().system.partitions))(self.check))

    def test_have_gpu_only(self):
        self.assertTrue((filters.have_gpu_only())(self.check))

    def test_have_cpu_only(self):
        self.assertFalse((filters.have_cpu_only())(self.check))
