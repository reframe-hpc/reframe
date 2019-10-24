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
    def expected_cmd_mount_points(self):
        pass

    @property
    @abc.abstractmethod
    def expected_cmd_prepare(self):
        pass

    @property
    @abc.abstractmethod
    def expected_cmd_with_run_opts(self):
        pass

    def setUp(self):
        self.container_platform = self.create_container_platform()

    def test_mount_points(self):
        self.container_platform.image = 'name:tag'
        self.container_platform.mount_points = [('/path/one', '/one'),
                                                ('/path/two', '/two')]
        self.container_platform.commands = ['cmd1', 'cmd2']
        self.container_platform.workdir = '/stagedir'
        assert (self.expected_cmd_mount_points ==
                self.container_platform.launch_command())

    def test_missing_image(self):
        self.container_platform.commands = ['cmd']
        with pytest.raises(ContainerError):
            self.container_platform.validate()

    def test_missing_commands(self):
        self.container_platform.image = 'name:tag'
        with pytest.raises(ContainerError):
            self.container_platform.validate()

    def test_prepare_command(self):
        self.container_platform.image = 'name:tag'
        assert (self.expected_cmd_prepare ==
                self.container_platform.emit_prepare_commands())

    def test_run_opts(self):
        self.container_platform.image = 'name:tag'
        self.container_platform.commands = ['cmd']
        self.container_platform.mount_points = [('/path/one', '/one')]
        self.container_platform.workdir = '/stagedir'
        assert (self.expected_cmd_with_run_opts ==
                self.container_platform.launch_command())


class TestDocker(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return containers.Docker()

    @property
    def expected_cmd_mount_points(self):
        return ('docker run --rm -v "/path/one":"/one" -v "/path/two":"/two" '
                "name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def expected_cmd_prepare(self):
        return []

    @property
    def expected_cmd_with_run_opts(self):
        return ('docker run --rm -v "/path/one":"/one" '
                "name:tag bash -c 'cd /stagedir; cmd'")


class TestShifterNG(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return containers.ShifterNG()

    @property
    def expected_cmd_mount_points(self):
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def expected_cmd_prepare(self):
        return ['shifter pull name:tag']

    @property
    def expected_cmd_with_run_opts(self):
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "name:tag bash -c 'cd /stagedir; cmd'")


class TestShifterNGWithMPI(TestShifterNG):
    def create_container_platform(self):
        ret = containers.ShifterNG()
        ret.with_mpi = True
        return ret

    @property
    def expected_cmd_mount_points(self):
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "--mpi name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def expected_cmd_with_run_opts(self):
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--mpi name:tag bash -c 'cd /stagedir; cmd'")


class TestSarus(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return containers.Sarus()

    @property
    def expected_cmd_mount_points(self):
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def expected_cmd_prepare(self):
        return ['sarus pull name:tag']

    @property
    def expected_cmd_with_run_opts(self):
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "name:tag bash -c 'cd /stagedir; cmd'")


class TestSarusWithMPI(TestSarus):
    def create_container_platform(self):
        ret = containers.Sarus()
        ret.with_mpi = True
        return ret

    @property
    def expected_cmd_mount_points(self):
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "--mpi name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def expected_cmd_with_run_opts(self):
        self.container_platform.with_mpi = True
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--mpi name:tag bash -c 'cd /stagedir; cmd'")


class TestSingularity(_ContainerPlatformTest, unittest.TestCase):
    def create_container_platform(self):
        return containers.Singularity()

    @property
    def expected_cmd_mount_points(self):
        return ('singularity exec -B"/path/one:/one" -B"/path/two:/two" '
                "name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def expected_cmd_prepare(self):
        return []

    @property
    def expected_cmd_with_run_opts(self):
        return ('singularity exec -B"/path/one:/one" '
                "name:tag bash -c 'cd /stagedir; cmd'")


class TestSingularityWithCuda(TestSingularity):
    def create_container_platform(self):
        ret = containers.Singularity()
        ret.with_cuda = True
        return ret

    @property
    def expected_cmd_mount_points(self):
        return ('singularity exec -B"/path/one:/one" -B"/path/two:/two" '
                "--nv name:tag bash -c 'cd /stagedir; cmd1; cmd2'")

    @property
    def expected_cmd_prepare(self):
        return []

    @property
    def expected_cmd_with_run_opts(self):
        self.container_platform.with_cuda = True
        return ('singularity exec -B"/path/one:/one" '
                "--nv name:tag bash -c 'cd /stagedir; cmd'")
