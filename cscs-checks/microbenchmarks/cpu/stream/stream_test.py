# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from hpctestlib.microbenchmarks.cpu.stream import Stream


@rfm.simple_test
class stream_check(Stream):
    valid_systems = ['dom:mc']
    valid_prog_environs = ['PrgEnv-cray']
    num_tasks = 2
    num_cpus_per_task = 36
