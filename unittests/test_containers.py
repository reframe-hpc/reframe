import abc
import unittest

import pytest
import reframe.core.containers as containers
from reframe.core.exceptions import ContainerError


class _ContainerPlatformTest(abc.ABC):
    @abc.abstractmethod
    def create_container_platform(self):
        pass

    @property
    @abc.abstractmethod
    def exp_cmd_mount_points(self):
        pass

    @property
    @abc.abstractmethod
    def exp_cmd_custom_registry(self):
        pass

    def setUp(self):
        self.container_platform = self.create_container_platform()

    def test_mount_points(self):
        self.container_platform.image = 'name:tag'
        self.container_platform.mount_points = [('/path/one', '/one'),
                                                ('/path/two', '/two')]
        self.container_platform.commands = ['cmd1', 'cmd2']
        self.container_platform.workdir = '/stagedir'
        assert (self.exp_cmd_mount_points ==
                self.container_platform.emit_launch_cmds())

    def test_missing_image(self):
        self.container_platform.commands = ['cmd']
        with pytest.raises(ContainerError):
            self.container_platform.validate()

    def test_missing_commands(self):
        self.container_platform.image = 'name:tag'
        with pytest.raises(ContainerError):
            self.container_platform.validate()

    def test_custom_registry(self):
        self.container_platform.registry = 'registry/custom'
        self.container_platform.image = 'name:tag'
        self.container_platform.commands = ['cmd']
        self.container_platform.mount_points = [('/path/one', '/one')]
        self.container_platform.workdir = '/stagedir'
        assert (self.exp_cmd_custom_registry ==
                self.container_platform.emit_launch_cmds())


class TestDocker(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return containers.Docker()

    @property
    def exp_cmd_mount_points(self):
        return ('docker run --rm -v "/path/one":"/one" -v "/path/two":"/two" '
                "name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def exp_cmd_custom_registry(self):
        return ('docker run --rm -v "/path/one":"/one" '
                'registry/custom/name:tag '
                "bash -c 'cd /stagedir; cmd'")


class TestShifterNG(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return containers.ShifterNG()

    @property
    def exp_cmd_mount_points(self):
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def exp_cmd_custom_registry(self):
        self.container_platform.with_mpi = True
        return ('shifter run --mpi '
                '--mount=type=bind,source="/path/one",destination="/one" '
                'registry/custom/name:tag '
                "bash -c 'cd /stagedir; cmd'")


class TestSingularity(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return containers.Singularity()

    @property
    def exp_cmd_mount_points(self):
        return ('singularity exec -B"/path/one","/one" -B"/path/two","/two" '
                "name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def exp_cmd_custom_registry(self):
        self.container_platform.with_cuda = True
        return ('singularity exec --nv -B"/path/one","/one" '
                'registry/custom/name:tag '
                "bash -c 'cd /stagedir; cmd'")
