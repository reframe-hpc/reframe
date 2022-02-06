# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import pytest

import reframe.frontend.autodetect as autodetect
import unittests.utility as test_util
from reframe.core.runtime import runtime
from reframe.utility.cpuinfo import cpuinfo


@pytest.fixture
def temp_topo(tmp_path, monkeypatch):
    # Monkey-patch HOME, since topology is always written there
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(autodetect, '_TREAT_WARNINGS_AS_ERRORS', True)

    # Create a devices file manually, since it is not auto-generated
    meta_prefix = tmp_path / '.reframe' / 'topology' / 'generic-default'
    os.makedirs(meta_prefix)
    with open(meta_prefix / 'devices.json', 'w') as fp:
        json.dump([
            {
                'type': 'gpu',
                'arch': 'a100',
                'num_devices': 8
            }
        ], fp)


@pytest.fixture
def invalid_topo(tmp_path, monkeypatch):
    # Monkey-patch HOME, since topology is always written there
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(autodetect, '_TREAT_WARNINGS_AS_ERRORS', True)

    # Create invalid processor and devices files
    meta_prefix = tmp_path / '.reframe' / 'topology' / 'generic-default'
    os.makedirs(meta_prefix)
    with open(meta_prefix / 'processor.json', 'w') as fp:
        fp.write('{')

    with open(meta_prefix / 'devices.json', 'w') as fp:
        fp.write('{')


@pytest.fixture
def default_exec_ctx(make_exec_ctx_g, temp_topo):
    yield from make_exec_ctx_g()


@pytest.fixture
def remote_exec_ctx(make_exec_ctx, temp_topo):
    if test_util.USER_CONFIG_FILE is None:
        pytest.skip('no user configuration file supplied')

    ctx = make_exec_ctx(test_util.USER_CONFIG_FILE,
                        test_util.USER_SYSTEM,
                        {'general/remote_detect': True})
    yield ctx


@pytest.fixture
def invalid_topo_exec_ctx(make_exec_ctx_g, invalid_topo):
    yield from make_exec_ctx_g()


def test_autotect(default_exec_ctx):
    autodetect.detect_topology()
    part = runtime().system.partitions[0]
    assert part.processor.info == cpuinfo()
    if part.processor.info:
        assert part.processor.num_cpus == part.processor.info['num_cpus']

    assert len(part.devices) == 1
    assert part.devices[0].info == {
        'type': 'gpu',
        'arch': 'a100',
        'num_devices': 8
    }
    assert part.devices[0].device_type == 'gpu'

    # Test immutability of ProcessorInfo and DeviceInfo
    with pytest.raises(AttributeError):
        part.processor.num_cpus = 3

    with pytest.raises(AttributeError):
        part.processor.foo = 10

    with pytest.raises(AttributeError):
        part.devices[0].arch = 'foo'

    with pytest.raises(AttributeError):
        part.devices[0].foo = 10


def test_autotect_with_invalid_files(invalid_topo_exec_ctx):
    autodetect.detect_topology()
    part = runtime().system.partitions[0]
    assert part.processor.info == cpuinfo()
    assert part.devices == []


def test_remote_autodetect(remote_exec_ctx):
    # All we can do with this test is to trigger the remote auto-detection
    # path; since we don't know what the remote user system is, we cannot test
    # if the topology is right.
    partition = test_util.partition_by_scheduler()
    if not partition:
        pytest.skip('job submission not supported')

    autodetect.detect_topology()
