# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.containers as containers
from reframe.core.exceptions import ContainerError


@pytest.fixture(params=[
    'Docker', 'Docker+nocommand', 'Docker+nopull', 'Sarus', 'Sarus+nocommand',
    'Sarus+nopull', 'Sarus+custompull', 'Sarus+mpi', 'Shifter', 'Shifter+mpi',
    'Singularity', 'Singularity+cuda', 'Singularity+nocommand'
])
def container_variant(request):
    return request.param


@pytest.fixture
def container_platform(container_variant):
    name = container_variant.split('+')[0]
    ret = containers.__dict__[name]()
    if '+nocommand' not in container_variant:
        ret.command = 'cmd'

    if '+mpi' in container_variant:
        ret.with_mpi = True

    if '+cuda' in container_variant:
        ret.with_cuda = True

    if container_variant == 'Sarus+custompull':
        ret.image = 'load/library/image:tag'
    else:
        ret.image = 'image:tag'

    if '+nopull' in container_variant:
        ret.pull_command = None

    if container_variant == 'Sarus+custompull':
        ret.pull_command = ('docker pull registry/image:tag; '
                            'docker save -o local_image.tar; '
                            'sarus load local_image.tar image:tag')

    return ret


@pytest.fixture
def expected_cmd_mount_points(container_variant):
    if container_variant in {'Docker', 'Docker+nopull'}:
        return ('docker run --rm --workdir="/stagedir" -v "/path/one":"/one" '
                '-v "/path/two":"/two" image:tag cmd')
    elif container_variant == 'Docker+nocommand':
        return ('docker run --rm --workdir="/stagedir" -v "/path/one":"/one" '
                '-v "/path/two":"/two" image:tag')
    elif container_variant in {'Sarus', 'Sarus+nopull'}:
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                'image:tag cmd')
    elif container_variant == 'Sarus+custompull':
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                'load/library/image:tag cmd')
    elif container_variant == 'Sarus+nocommand':
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                'image:tag')
    elif container_variant == 'Sarus+mpi':
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mpi image:tag cmd')
    elif container_variant == 'Singularity':
        return ('singularity exec --workdir="/stagedir" -B"/path/one:/one" '
                '-B"/path/two:/two" image:tag cmd')
    elif container_variant == 'Singularity+cuda':
        return ('singularity exec --workdir="/stagedir" -B"/path/one:/one" '
                '-B"/path/two:/two" '
                '--nv image:tag cmd')
    elif container_variant == 'Singularity+nocommand':
        return ('singularity run --workdir="/stagedir" -B"/path/one:/one" '
                '-B"/path/two:/two" image:tag')
    elif container_variant == 'Shifter':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "image:tag bash -c 'cd /stagedir;cmd'")
    elif container_variant == 'Shifter+mpi':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                "--mpi image:tag bash -c 'cd /stagedir;cmd'")


@pytest.fixture
def expected_cmd_prepare(container_variant):
    if container_variant in ('Docker', 'Docker+nocommand'):
        return ['docker pull image:tag']
    if container_variant in ('Shifter', 'Shifter+mpi'):
        return ['shifter pull image:tag']
    elif container_variant in ('Sarus', 'Sarus+nocommand', 'Sarus+mpi'):
        return ['sarus pull image:tag']
    elif container_variant == 'Sarus+custompull':
        return [
            'docker pull registry/image:tag; '
            'docker save -o local_image.tar; '
            'sarus load local_image.tar image:tag'
        ]
    else:
        return []


@pytest.fixture
def expected_cmd_run_opts(container_variant):
    if container_variant in {'Docker', 'Docker+nopull'}:
        return ('docker run --rm --workdir="/stagedir" -v "/path/one":"/one" '
                '--foo --bar image:tag cmd')
    if container_variant == 'Docker+nocommand':
        return ('docker run --rm --workdir="/stagedir" -v "/path/one":"/one" '
                '--foo --bar image:tag')
    elif container_variant == 'Shifter':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--foo --bar image:tag bash -c 'cd /stagedir;cmd'")
    elif container_variant == 'Shifter+mpi':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                "--mpi --foo --bar image:tag bash -c 'cd /stagedir;cmd'")
    elif container_variant in {'Sarus', 'Sarus+nopull'}:
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--foo --bar image:tag cmd')
    elif container_variant == 'Sarus+nocommand':
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--foo --bar image:tag')
    elif container_variant == 'Sarus+mpi':
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mpi --foo --bar image:tag cmd')
    elif container_variant == 'Sarus+custompull':
        return ('sarus run --workdir="/stagedir" '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--foo --bar load/library/image:tag cmd')
    elif container_variant == 'Singularity':
        return ('singularity exec --workdir="/stagedir" -B"/path/one:/one" '
                '--foo --bar image:tag cmd')
    elif container_variant == 'Singularity+cuda':
        return ('singularity exec --workdir="/stagedir" -B"/path/one:/one" '
                '--nv --foo --bar image:tag cmd')
    elif container_variant == 'Singularity+nocommand':
        return ('singularity run --workdir="/stagedir" -B"/path/one:/one" '
                '--foo --bar image:tag')


def test_mount_points(container_platform, expected_cmd_mount_points):
    container_platform.mount_points = [('/path/one', '/one'),
                                       ('/path/two', '/two')]
    container_platform.workdir = '/stagedir'
    assert container_platform.launch_command() == expected_cmd_mount_points


def test_missing_image(container_platform):
    container_platform.image = None
    with pytest.raises(ContainerError):
        container_platform.validate()


def test_prepare_command(container_platform, expected_cmd_prepare):
    assert container_platform.emit_prepare_commands() == expected_cmd_prepare


def test_run_opts(container_platform, expected_cmd_run_opts):
    container_platform.mount_points = [('/path/one', '/one')]
    container_platform.workdir = '/stagedir'
    container_platform.options = ['--foo', '--bar']
    assert container_platform.launch_command() == expected_cmd_run_opts
