# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import time

import reframe.utility.profile as prof


def test_region_profiling():
    profiler = prof.TimeProfiler()
    profiler.enter_region('forloop')
    for _ in range(10):
        profiler.enter_region('sleep')
        time.sleep(.1)
        profiler.exit_region()

    profiler.exit_region()

    t_sleep = profiler.total_time('sleep')
    t_forloop = profiler.total_time('forloop')
    assert t_sleep >= 1
    assert t_forloop > t_sleep

    with pytest.raises(prof.ProfilerError):
        profiler.total_time('foo')

    profiler.enter_region('unexited')
    with pytest.raises(prof.ProfilerError):
        profiler.total_time('unexited')


def test_time_region_ctxmgr():
    with prof.time_region('forloop') as profiler:
        for _ in range(10):
            with prof.time_region('sleep', profiler):
                time.sleep(.1)

    t_sleep = profiler.total_time('sleep')
    t_forloop = profiler.total_time('forloop')
    assert t_sleep >= 1
    assert t_forloop > t_sleep


def test_time_region_profiler_ctxmgr():
    profiler = prof.TimeProfiler()
    with profiler.time_region('forloop'):
        for _ in range(10):
            with profiler.time_region('sleep'):
                time.sleep(.1)

    t_sleep = profiler.total_time('sleep')
    t_forloop = profiler.total_time('forloop')
    assert t_sleep >= 1
    assert t_forloop > t_sleep
