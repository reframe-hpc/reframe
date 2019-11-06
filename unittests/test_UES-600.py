import itertools
import os
import unittest
from tempfile import NamedTemporaryFile

import reframe.utility.sanity as sn
from reframe.core.deferrable import evaluate, make_deferrable
from reframe.core.exceptions import SanityError
from unittests.fixtures import TEST_RESOURCES_CHECKS


class TestPatternMatchingFunctions(unittest.TestCase):
    def test_extract_patrun(self):
        self.rpt = 'eff.patrun'
        regex0 = r'..include.sph.momentumAndEnergyIAD.hpp\n\|{4}-*(\n.*)*\|{4}===='
        regex1 = r'(\d\|{3}\s+(?P<pctg>\S+)%\s+\|\s+\S+\s+\|\s+--\s+\|\s+--\s+\|\s+line.\d+\n)'
        res = evaluate(
                sn.extractsingle(regex1,
                    evaluate(sn.extractsingle(regex0, self.rpt)),
                    'pctg', float)
                )

        f=open("eff.patrun.rpt", "+w")
        f.write("res={} {}\n".format(res, type(res)))
        f.close()

