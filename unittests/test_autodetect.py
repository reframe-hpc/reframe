# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import pytest


from reframe.core.runtime import runtime
from reframe.frontend.autodetect import detect_topology
from reframe.utility.cpuinfo import cpuinfo


@pytest.fixture
def exec_ctx(make_exec_ctx_g, tmp_path, monkeypatch):
    # Monkey-patch HOME, since topology is always written there
    monkeypatch.setenv('HOME', str(tmp_path))

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

    yield from make_exec_ctx_g()


@pytest.fixture
def invalid_topology_exec_ctx(make_exec_ctx_g, tmp_path, monkeypatch):
    # Monkey-patch HOME, since topology is always written there
    monkeypatch.setenv('HOME', str(tmp_path))

    # Create invalid processor and devices files
    meta_prefix = tmp_path / '.reframe' / 'topology' / 'generic-default'
    os.makedirs(meta_prefix)
    with open(meta_prefix / 'processor.json', 'w') as fp:
        fp.write('{')

    with open(meta_prefix / 'devices.json', 'w') as fp:
        fp.write('{')

    yield from make_exec_ctx_g()


def test_autotect(exec_ctx):
    detect_topology()
    part = runtime().system.partitions[0]
    assert part.processor.info == cpuinfo()
    assert part.devices == [{'type': 'gpu', 'arch': 'a100', 'num_devices': 8}]


def test_autotect_with_invalid_files(invalid_topology_exec_ctx):
    detect_topology()
    part = runtime().system.partitions[0]
    assert part.processor.info == cpuinfo()
    assert part.devices == []
