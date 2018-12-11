import unittest

import reframe
from reframe.frontend.check_filters import *
from reframe.core.pipeline import RegressionTest

class TestCheckFilters(unittest.TestCase):
    def setUp(self):
        #TODO: use correct initialization
        self.check = RegressionTest()
        self.check.name = "check1"
        self.check.tags = {'a', 'b', 'c', 'd'} 
        self.check.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        self.check.valid_systems = ['daint:gpu', 'daint:mc',
            'dom:gpu', 'dom:mc', 'kesch:cn', 'kesch:pn']
        self.check.num_gpus_per_node = 1

    def test_have_name(self):
        self.assertTrue((have_name('check1'))(self.check))
        self.assertFalse((have_name('check2'))(self.check))

    def test_have_not_name(self):
        self.assertTrue((have_not_name('check2'))(self.check))
        self.assertFalse((have_not_name('check1'))(self.check))
              
    def test_have_tags(self):
        self.assertTrue((have_tag(['a', 'c']))(self.check)) 
        self.assertFalse((have_tag(['a', 'b', 'c', 'd', 'e']))(self.check)) 
        self.assertFalse((have_tag(['x', 'y', 'z']))(self.check)) 

    def test_have_prgenv(self):
        self.assertTrue((have_prgenv(['PrgEnv-cray', 'PrgEnv-gnu'])))
        self.assertFalse((have_prgenv(['PrgEnv-pgi'])))
