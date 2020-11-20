# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Torque backend
#
# - Initial version submitted by Samuel Moors, Vrije Universiteit Brussel (VUB)
#

import re
import os
import time

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobError, JobSchedulerError
from reframe.core.schedulers.pbs import PbsJobScheduler, _run_strict


@register_scheduler('torque')
class TorqueJobScheduler(PbsJobScheduler):
    TASKS_OPT = '-l nodes={num_nodes}:ppn={num_cpus_per_node}'
