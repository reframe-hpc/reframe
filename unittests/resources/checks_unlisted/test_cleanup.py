import unittest
from unittests.test_cli import TestFrontend


class TestCleanup(TestFrontend):
    def test_dependency_cli(self):
        self.checkpath =
        ['unittests/resources/checks_unlisted/cleanup_checks.py']
        self.action = 'run'
        self.more_options = ['-n', 'Dependency']
        returncode, stdout, _ = self._run_reframe()
        self.assertIn('Running 4 check(s)', stdout)
        self.assertEqual(0, returncode)

    def test_multi_dependency_cli(self):
        self.checkpath =
        ['unittests/resources/checks_unlisted/cleanup_checks.py']
        self.action = 'run'
        self.more_options = ['-n', 'MultiDependency']
        returncode, stdout, _ = self._run_reframe()
        self.assertIn('Running 7 check(s)', stdout)
        self.assertEqual(0, returncode)
