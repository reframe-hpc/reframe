# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.containers as containers
import unittests.fixtures as fixtures
from reframe.core.exceptions import ContainerError


@pytest.fixture(params=[
    'Docker',
    'Sarus', 'Sarus+mpi', 'Sarus+localimage',
    'Shifter', 'Shifter+mpi', 'Shifter+localimage',
    'Singularity', 'Singularity+cuda'
])
def container_variant(request):
    return request.param


@pytest.fixture
def container_platform(container_variant):
    name = container_variant.split('+')[0]
    ret = containers.__dict__[name]()
    if '+mpi' in container_variant:
        ret.with_mpi = True

    if '+cuda' in container_variant:
        ret.with_cuda = True

    if '+localimage' in container_variant:
        ret.image = 'load/library/image:tag'
    else:
        ret.image = 'image:tag'

    return ret


@pytest.fixture
def expected_cmd_mount_points(container_variant):
    if container_variant == 'Docker':
        return ('docker run --rm -v "/path/one":"/one" -v "/path/two":"/two" '
                "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Sarus':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Sarus+mpi':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "--mpi image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Sarus+localimage':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "load/library/image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Singularity':
        return ('singularity exec -B"/path/one:/one" -B"/path/two:/two" '
                "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Singularity+cuda':
        return ('singularity exec -B"/path/one:/one" -B"/path/two:/two" '
                "--nv image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Shifter':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Shifter+localimage':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "load/library/image:tag bash -c 'cd /stagedir; cmd1; cmd2'")
    elif container_variant == 'Shifter+mpi':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "--mpi image:tag bash -c 'cd /stagedir; cmd1; cmd2'")


@pytest.fixture
def expected_cmd_prepare(container_variant):
    if container_variant in ('Shifter', 'Shifter+mpi'):
        return ['shifter pull image:tag']
    elif container_variant in ('Sarus', 'Sarus+mpi'):
        return ['sarus pull image:tag']
    else:
        return []


@pytest.fixture
def expected_cmd_run_opts(container_variant):
    if container_variant == 'Docker':
        return ('docker run --rm -v "/path/one":"/one" --foo --bar '
                "image:tag bash -c 'cd /stagedir; cmd'")
    elif container_variant == 'Shifter':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--foo --bar image:tag bash -c 'cd /stagedir; cmd'")
    elif container_variant == 'Shifter+mpi':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--mpi --foo --bar image:tag bash -c 'cd /stagedir; cmd'")
    elif container_variant == 'Shifter+localimage':
        return (
            'shifter run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            "--foo --bar load/library/image:tag bash -c 'cd /stagedir; cmd'"
        )
    elif container_variant == 'Sarus':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--foo --bar image:tag bash -c 'cd /stagedir; cmd'")
    elif container_variant == 'Sarus+mpi':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--mpi --foo --bar image:tag bash -c 'cd /stagedir; cmd'")
    elif container_variant == 'Sarus+localimage':
        return (
            'sarus run '
            '--mount=type=bind,source="/path/one",destination="/one" '
            "--foo --bar load/library/image:tag bash -c 'cd /stagedir; cmd'"
        )
    elif container_variant == 'Singularity':
        return ('singularity exec -B"/path/one:/one" '
                "--foo --bar image:tag bash -c 'cd /stagedir; cmd'")
    elif container_variant == 'Singularity+cuda':
        return ('singularity exec -B"/path/one:/one" '
                "--nv --foo --bar image:tag bash -c 'cd /stagedir; cmd'")


def test_mount_points(container_platform, expected_cmd_mount_points):
    container_platform.mount_points = [('/path/one', '/one'),
                                       ('/path/two', '/two')]
    container_platform.commands = ['cmd1', 'cmd2']
    container_platform.workdir = '/stagedir'
    assert container_platform.launch_command() == expected_cmd_mount_points


def test_missing_image(container_platform):
    container_platform.image = None
    container_platform.commands = ['cmd']
    with pytest.raises(ContainerError):
        container_platform.validate()


def test_missing_commands(container_platform):
    container_platform.image = 'image:tag'
    with pytest.raises(ContainerError):
        container_platform.validate()


def test_prepare_command(container_platform, expected_cmd_prepare):
    assert container_platform.emit_prepare_commands() == expected_cmd_prepare


def test_run_opts(container_platform, expected_cmd_run_opts):
    container_platform.commands = ['cmd']
    container_platform.mount_points = [('/path/one', '/one')]
    container_platform.workdir = '/stagedir'
    container_platform.options = ['--foo', '--bar']
    assert container_platform.launch_command() == expected_cmd_run_opts
