# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import pytest

import reframe.core.containers as containers
from reframe.core.exceptions import ContainerError


@pytest.fixture(params=['docker', 'shifter', 'shifter_localimage',
                        'shifter_mpi', 'sarus', 'sarus_localimage',
                        'sarus_mpi', 'singularity',
                        'singularity_cuda'])
def container_platform(request):
    platform = request.param.split('_')[0]
    if platform == 'docker':
        return containers.Docker(), request.param
    elif platform == 'shifter':
        ret = containers.Shifter()
        if 'mpi' in request.param:
            ret.with_mpi = True

        return ret, request.param
    elif platform == 'sarus':
        ret = containers.Sarus()
        if 'mpi' in request.param:
            ret.with_mpi = True

        return ret, request.param
    else:
        ret = containers.Singularity()
        if 'cuda' in request.param:
            ret.with_cuda = True

        return ret, request.param


def _expected_docker_cmd_prepare():
    return []


def _expected_shifter_cmd_prepare():
    return ['shifter pull image:tag']


def _expected_sarus_cmd_prepare():
    return ['sarus pull image:tag']


_expected_shifter_localimage_cmd_prepare = _expected_docker_cmd_prepare
_expected_shifter_mpi_cmd_prepare = _expected_shifter_cmd_prepare
_expected_sarus_localimage_cmd_prepare = _expected_docker_cmd_prepare
_expected_sarus_mpi_cmd_prepare = _expected_sarus_cmd_prepare
_expected_singularity_cmd_prepare = _expected_docker_cmd_prepare
_expected_singularity_cuda_cmd_prepare = _expected_singularity_cmd_prepare


@pytest.fixture
def expected_cmd_prepare(container_platform):
    return globals()[f'_expected_{container_platform[1]}_cmd_prepare']()


def _expected_docker_cmd_mount_points():
    return ('docker run --rm -v "/path/one":"/one" -v "/path/two":"/two" '
            "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


def _expected_shifter_cmd_mount_points():
    return ('shifter run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            '--mount=type=bind,source="/path/two",destination="/two" '
            "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


def _expected_shifter_mpi_cmd_mount_points():
    return ('shifter run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            '--mount=type=bind,source="/path/two",destination="/two" '
            "--mpi image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


def _expected_sarus_cmd_mount_points():
    return ('sarus run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            '--mount=type=bind,source="/path/two",destination="/two" '
            "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


def _expected_sarus_mpi_cmd_mount_points():
    return ('sarus run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            '--mount=type=bind,source="/path/two",destination="/two" '
            "--mpi image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


def _expected_singularity_cmd_mount_points():
    return ('singularity exec -B"/path/one:/one" -B"/path/two:/two" '
            "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


def _expected_singularity_cuda_cmd_mount_points():
    return ('singularity exec -B"/path/one:/one" -B"/path/two:/two" '
            "--nv image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


_expected_shifter_localimage_cmd_mount_points = (
    _expected_shifter_cmd_mount_points
)
_expected_sarus_localimage_cmd_mount_points = _expected_sarus_cmd_mount_points


@pytest.fixture
def expected_cmd_mount_points(container_platform):
    return globals()[f'_expected_{container_platform[1]}_cmd_mount_points']()


def _expected_docker_cmd_with_run_opts():
    return ('docker run --rm -v "/path/one":"/one" --foo --bar '
            "image:tag bash -c 'cd /stagedir; cmd'")


def _expected_shifter_cmd_with_run_opts():
    return ('shifter run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            "--foo --bar image:tag bash -c 'cd /stagedir; cmd'")


def _expected_shifter_mpi_cmd_with_run_opts():
    return ('shifter run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            "--mpi --foo --bar image:tag bash -c 'cd /stagedir; cmd'")


def _expected_sarus_cmd_with_run_opts():
    return ('sarus run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            "--foo --bar image:tag bash -c 'cd /stagedir; cmd'")


def _expected_sarus_mpi_cmd_with_run_opts():
    return ('sarus run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            "--mpi --foo --bar image:tag bash -c 'cd /stagedir; cmd'")


def _expected_singularity_cmd_with_run_opts():
    return ('singularity exec -B"/path/one:/one" '
            "--foo --bar image:tag bash -c 'cd /stagedir; cmd'")


def _expected_singularity_cuda_cmd_with_run_opts():
    return ('singularity exec -B"/path/one:/one" '
            "--nv --foo --bar image:tag bash -c 'cd /stagedir; cmd'")


_expected_shifter_localimage_cmd_with_run_opts = (
    _expected_shifter_cmd_with_run_opts
)
_expected_sarus_localimage_cmd_with_run_opts = (
    _expected_sarus_cmd_with_run_opts
)

@pytest.fixture
def expected_cmd_with_run_opts(container_platform):
    return globals()[f'_expected_{container_platform[1]}_cmd_with_run_opts']()


def test_mount_points(container_platform, expected_cmd_mount_points):
    platform = container_platform[0]
    platform.image = 'image:tag'
    platform.mount_points = [('/path/one', '/one'), ('/path/two', '/two')]
    platform.commands = ['cmd1', 'cmd2']
    platform.workdir = '/stagedir'
    assert expected_cmd_mount_points == platform.launch_command()


def test_missing_image(container_platform):
    platform = container_platform[0]
    platform.commands = ['cmd']
    with pytest.raises(ContainerError):
        platform.validate()


def test_missing_commands(container_platform):
    platform = container_platform[0]
    platform.image = 'image:tag'
    with pytest.raises(ContainerError):
        platform.validate()


def test_prepare_command(container_platform, expected_cmd_prepare):
    platform = container_platform[0]
    if container_platform[1] in {'shifter_localimage', 'sarus_localimage'}:
        platform.image = 'load/library/image:tag'
    else:
        platform.image = 'image:tag'

    assert expected_cmd_prepare == platform.emit_prepare_commands()


def test_run_opts(container_platform, expected_cmd_with_run_opts):
    platform = container_platform[0]
    platform.image = 'image:tag'
    platform.commands = ['cmd']
    platform.mount_points = [('/path/one', '/one')]
    platform.workdir = '/stagedir'
    platform.options = ['--foo', '--bar']
    assert expected_cmd_with_run_opts == platform.launch_command()
