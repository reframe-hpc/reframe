import abc
import unittest

import reframe.core.containerplatform as cp
from reframe.core.exceptions import ContainerPlatformError


class _ContainerPlatformTest:
    @abc.abstractmethod
    def create_container_platform(self):
        pass

    def setUp(self):
        self.container_platform = self.create_container_platform()


class TestDocker(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return cp.Docker()

    def test_mount_points(self):
        self.container_platform.image = 'name:tag'
        self.container_platform.mount_points = [('/path/one', '/one'),
                                                ('/path/two', '/two')]
        self.container_platform.commands = ['cmd1', 'cmd2']
        expected = [
            "docker run -v /path/one:/one -v /path/two:/two -v $PWD:/stagedir "
            "name:tag bash -c 'cd /stagedir; cmd1; cmd2'"
        ]
        self.assertEqual(expected,
                         self.container_platform.emit_launch_commands())

    def test_missing_image(self):
        self.container_platform.commands = ['cmd']
        self.assertRaises(ContainerPlatformError,
                          self.container_platform.emit_launch_commands)

    def test_missing_commands(self):
        self.container_platform.image = 'name:tag'
        self.assertRaises(ContainerPlatformError,
                          self.container_platform.emit_launch_commands)

    def test_custom_registry(self):
        self.container_platform.registry = 'registry/custom'
        self.container_platform.image = 'name:tag'
        self.container_platform.commands = ['cmd']
        expected = [
            "docker run -v $PWD:/stagedir registry/custom/name:tag "
            "bash -c 'cd /stagedir; cmd'"
        ]
        self.assertEqual(expected,
                         self.container_platform.emit_launch_commands())
