# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import pytest
import shutil


from reframe.core.runtime import runtime
from reframe.frontend.autodetect import detect_topology
from reframe.utility.cpuinfo import cpuinfo


@pytest.fixture
def exec_ctx(make_exec_ctx_g, tmp_path):
    # Copy the default settings to the temp dir
    shutil.copy('reframe/core/settings.py', tmp_path / 'settings.py')

    # Create a devices file manually, since it is not auto-generated
    meta_prefix = tmp_path / '_meta' / 'generic-default'
    os.makedirs(meta_prefix)
    with open(meta_prefix / 'devices.json', 'w') as fp:
        json.dump([
            {
                'type': 'gpu',
                'arch': 'a100',
                'num_devices': 8
            }
        ], fp)

    yield from make_exec_ctx_g(tmp_path / 'settings.py')


def test_autotect(exec_ctx):
    detect_topology()
    part = runtime().system.partitions[0]
    assert part.processor.info == cpuinfo()
    assert part.devices == [{'type': 'gpu', 'arch': 'a100', 'num_devices': 8}]
