# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.containers as containers
import reframe.core.warnings as warn
from reframe.core.exceptions import ContainerError


@pytest.fixture(params=[
    'Docker', 'Docker+nocommand', 'Docker+nopull',
    'Sarus', 'Sarus+nocommand', 'Sarus+nopull', 'Sarus+mpi', 'Sarus+load',
    'Shifter', 'Shifter+nocommand', 'Shifter+mpi', 'Shifter+nopull',
    'Shifter+load',
    'Singularity', 'Singularity+nocommand', 'Singularity+cuda'
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

    if '+load' in container_variant:
        ret.image = 'load/library/image:tag'
    else:
        ret.image = 'image:tag'

    if '+nopull' in container_variant:
        ret.pull_image = False

    return ret


@pytest.fixture
def expected_cmd_mount_points(container_variant):
    if container_variant in {'Docker', 'Docker+nopull'}:
        return ('docker run --rm -v "/path/one":"/one" '
                '-v "/path/two":"/two" '
                '-v "/foo":"/rfm_workdir" image:tag cmd')
    elif container_variant == 'Docker+nocommand':
        return ('docker run --rm -v "/path/one":"/one" '
                '-v "/path/two":"/two" '
                '-v "/foo":"/rfm_workdir" image:tag')
    elif container_variant in {'Sarus', 'Sarus+nopull'}:
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                'image:tag cmd')
    elif container_variant == 'Sarus+nocommand':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                'image:tag')
    elif container_variant == 'Sarus+mpi':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--mpi image:tag cmd')
    elif container_variant == 'Sarus+load':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                'load/library/image:tag cmd')
    elif container_variant in {'Shifter', 'Shifter+nopull'}:
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                'image:tag cmd')
    elif container_variant == 'Shifter+nocommand':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                'image:tag')
    elif container_variant == 'Shifter+mpi':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--mpi image:tag cmd')
    elif container_variant == 'Shifter+load':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/path/two",destination="/two" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                'load/library/image:tag cmd')
    elif container_variant in {'Singularity', 'Singularity+nopull'}:
        return ('singularity exec -B"/path/one:/one" '
                '-B"/path/two:/two" -B"/foo:/rfm_workdir" image:tag cmd')
    elif container_variant == 'Singularity+cuda':
        return ('singularity exec -B"/path/one:/one" '
                '-B"/path/two:/two" -B"/foo:/rfm_workdir" --nv image:tag cmd')
    elif container_variant == 'Singularity+nocommand':
        return ('singularity run -B"/path/one:/one" '
                '-B"/path/two:/two" -B"/foo:/rfm_workdir" image:tag')


@pytest.fixture
def expected_cmd_prepare(container_variant):
    if container_variant in {'Docker', 'Docker+nocommand'}:
        return ['docker pull image:tag']
    elif container_variant in {'Shifter', 'Shifter+nocommand', 'Shifter+mpi'}:
        return ['shifter pull image:tag']
    elif container_variant in {'Sarus', 'Sarus+nocommand', 'Sarus+mpi'}:
        return ['sarus pull image:tag']
    else:
        return []


@pytest.fixture
def expected_cmd_run_opts(container_variant):
    if container_variant in {'Docker', 'Docker+nopull'}:
        return ('docker run --rm -v "/path/one":"/one" '
                '-v "/foo":"/rfm_workdir" --foo --bar image:tag cmd')
    if container_variant == 'Docker+nocommand':
        return ('docker run --rm -v "/path/one":"/one" '
                '-v "/foo":"/rfm_workdir" --foo --bar image:tag')
    elif container_variant in {'Shifter', 'Shifter+nopull'}:
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--foo --bar image:tag cmd')
    elif container_variant == 'Shifter+nocommand':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--foo --bar image:tag')
    elif container_variant == 'Shifter+mpi':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--mpi --foo --bar image:tag cmd')
    elif container_variant == 'Shifter+load':
        return ('shifter run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--foo --bar load/library/image:tag cmd')
    elif container_variant in {'Sarus', 'Sarus+nopull'}:
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--foo --bar image:tag cmd')
    elif container_variant == 'Sarus':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--foo --bar image:tag cmd')
    elif container_variant == 'Sarus+load':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--foo --bar load/library/image:tag cmd')
    elif container_variant == 'Sarus+nocommand':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--foo --bar image:tag')
    elif container_variant == 'Sarus+mpi':
        return ('sarus run '
                '--mount=type=bind,source="/path/one",destination="/one" '
                '--mount=type=bind,source="/foo",destination="/rfm_workdir" '
                '--mpi --foo --bar image:tag cmd')
    elif container_variant in {'Singularity'}:
        return ('singularity exec -B"/path/one:/one" -B"/foo:/rfm_workdir" '
                '--foo --bar image:tag cmd')
    elif container_variant == 'Singularity+cuda':
        return ('singularity exec -B"/path/one:/one" -B"/foo:/rfm_workdir" '
                '--nv --foo --bar image:tag cmd')
    elif container_variant == 'Singularity+nocommand':
        return ('singularity run -B"/path/one:/one" -B"/foo:/rfm_workdir" '
                '--foo --bar image:tag')


def test_mount_points(container_platform, expected_cmd_mount_points):
    container_platform.mount_points = [('/path/one', '/one'),
                                       ('/path/two', '/two')]
    cmd = container_platform.launch_command('/foo')
    assert cmd == expected_cmd_mount_points


def test_missing_image(container_platform):
    container_platform.image = None
    with pytest.raises(ContainerError):
        container_platform.validate()


def test_prepare_command(container_platform, expected_cmd_prepare):
    commands = container_platform.emit_prepare_commands('/foo')
    assert commands == expected_cmd_prepare


def test_run_opts(container_platform, expected_cmd_run_opts):
    container_platform.mount_points = [('/path/one', '/one')]
    container_platform.options = ['--foo', '--bar']
    assert container_platform.launch_command('/foo') == expected_cmd_run_opts


# Everything from this point is testing deprecated behavior

@pytest.fixture(params=['Docker', 'Singularity', 'Sarus', 'Shifter'])
def container_variant_noopt(request):
    return request.param


@pytest.fixture
def container_platform_noopt(container_variant_noopt):
    ret = containers.__dict__[container_variant_noopt]()
    ret.image = 'image:tag'
    ret.options = ['--foo']
    return ret


@pytest.fixture
def expected_run_with_commands(container_variant_noopt):
    if container_variant_noopt == 'Docker':
        return ("docker run --rm -v \"/foo\":\"/rfm_workdir\" "
                "--foo image:tag bash -c 'cd /rfm_workdir; cmd1; cmd2'")
    elif container_variant_noopt == 'Sarus':
        return (
            "sarus run "
            "--mount=type=bind,source=\"/foo\",destination=\"/rfm_workdir\" "
            "--foo image:tag bash -c 'cd /rfm_workdir; cmd1; cmd2'"
        )
    elif container_variant_noopt == 'Shifter':
        return (
            "shifter run "
            "--mount=type=bind,source=\"/foo\",destination=\"/rfm_workdir\" "
            "--foo image:tag bash -c 'cd /rfm_workdir; cmd1; cmd2'"
        )
    elif container_variant_noopt == 'Singularity':
        return ("singularity exec -B\"/foo:/rfm_workdir\" "
                "--foo image:tag bash -c 'cd /rfm_workdir; cmd1; cmd2'")


@pytest.fixture
def expected_run_with_workdir(container_variant_noopt):
    if container_variant_noopt == 'Docker':
        return ("docker run --rm -v \"/foo\":\"/rfm_workdir\" "
                "--foo image:tag bash -c 'cd foodir; cmd1; cmd2'")
    elif container_variant_noopt == 'Sarus':
        return (
            "sarus run "
            "--mount=type=bind,source=\"/foo\",destination=\"/rfm_workdir\" "
            "--foo image:tag bash -c 'cd foodir; cmd1; cmd2'"
        )
    elif container_variant_noopt == 'Shifter':
        return (
            "shifter run "
            "--mount=type=bind,source=\"/foo\",destination=\"/rfm_workdir\" "
            "--foo image:tag bash -c 'cd foodir; cmd1; cmd2'"
        )
    elif container_variant_noopt == 'Singularity':
        return ("singularity exec -B\"/foo:/rfm_workdir\" --foo image:tag "
                "bash -c 'cd foodir; cmd1; cmd2'")


def test_run_with_commands(container_platform_noopt,
                           expected_run_with_commands):
    with pytest.warns(warn.ReframeDeprecationWarning):
        container_platform_noopt.commands = ['cmd1', 'cmd2']

    found_commands = container_platform_noopt.launch_command('/foo')
    assert found_commands == expected_run_with_commands


def test_run_with_workdir(container_platform_noopt, expected_run_with_workdir):
    with pytest.warns(warn.ReframeDeprecationWarning):
        container_platform_noopt.commands = ['cmd1', 'cmd2']

    with pytest.warns(warn.ReframeDeprecationWarning):
        container_platform_noopt.workdir = 'foodir'

    found_commands = container_platform_noopt.launch_command('/foo')
    assert found_commands == expected_run_with_workdir
