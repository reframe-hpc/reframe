import unittest

import reframe.core.runtime as rt
import unittests.fixtures as fixtures


class TestRuntime(unittest.TestCase):
    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_hostsystem_api(self):
        system = rt.runtime().system
        self.assertEqual('testsys', system.name)
        self.assertEqual('Fake system for unit tests', system.descr)
        self.assertEqual(2, len(system.partitions))
        self.assertIsNotNone(system.partition('login'))
        self.assertIsNotNone(system.partition('gpu'))
        self.assertIsNone(system.partition('foobar'))

        # Test delegation to the underlying System
        self.assertEqual('.rfm_testing', system.prefix)
        self.assertEqual('.rfm_testing/resources', system.resourcesdir)
        self.assertEqual('.rfm_testing/perflogs', system.perflogdir)
