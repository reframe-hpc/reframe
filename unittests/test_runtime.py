import unittest

import reframe.core.runtime as rt
import unittests.fixtures as fixtures


class TestRuntime(unittest.TestCase):
    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_hostsystem_api(self):
        system = rt.runtime().system
        assert 'testsys' == system.name
        assert 'Fake system for unit tests' == system.descr
        assert 2 == len(system.partitions)
        assert system.partition('login') is not None
        assert system.partition('gpu') is not None
        assert system.partition('foobar') is None

        # Test delegation to the underlying System
        assert '.rfm_testing' == system.prefix
        assert '.rfm_testing/resources' == system.resourcesdir
        assert '.rfm_testing/perflogs' == system.perflogdir
