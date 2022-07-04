# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# A lightweight time profiler

import time
import sys

from collections import OrderedDict


class ProfilerError(Exception):
    pass


class time_region:
    '''Context manager for timing a code region'''

    def __init__(self, region, profiler=None):
        self._profiler = profiler or TimeProfiler()
        self._region = region

    def __enter__(self):
        self._profiler.enter_region(self._region)
        return self._profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._profiler.exit_region()


class TimeProfiler:
    def __init__(self):
        self._region_stack = ['root']
        if sys.version_info[:2] < (3, 8):
            self._region_times = OrderedDict()
        else:
            self._region_times = {}

    @property
    def current_region(self):
        return self._region_stack[-1]

    def enter_region(self, region_name):
        timestamp = time.time()
        region_fullname = f'{self.current_region}:{region_name}'
        if region_fullname in self._region_times:
            elapsed = self._region_times[region_fullname][1]
        else:
            elapsed = 0.0

        self._region_times[region_fullname] = (timestamp, elapsed)
        self._region_stack.append(region_fullname)

    def exit_region(self):
        timestamp = time.time()
        region = self.current_region
        t_start, elapsed = self._region_times[region]
        self._region_times[region] = (None, elapsed + timestamp - t_start)
        self._region_stack.pop()

    def total_time(self, region_name):
        for region in reversed(self._region_times.keys()):
            if (region == region_name or
                region.rsplit(':', maxsplit=1)[-1] == region_name):
                timestamp, elapsed = self._region_times[region]
                if timestamp:
                    raise ProfilerError(
                        f'region {region_name!r} has not exited'
                    )

                return elapsed

        raise ProfilerError(f'unknown region: {region_name!r}')

    def time_region(self, region):
        return globals()['time_region'](region, self)

    def print_report(self, print_fn=None):
        if print_fn is None:
            print_fn = print

        print_fn('>>> profiler report [start] <<<')
        for name, t_info in self._region_times.items():
            # Remove the root prefix
            levels = name.count(':')
            indent = ' '*4*(levels - 1)
            region_name = name.rsplit(':', maxsplit=1)[-1]
            msg = f'{indent}{region_name}: {t_info[1]:.6f} s'
            if t_info[0]:
                msg += ' <incomplete>'

            print_fn(msg)

        print_fn('>>> profiler report [ end ] <<<')
