# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.data_analytics.spark.spark_checks import compute_pi_check


@rfm.simple_test
class cscs_compute_pi_check(compute_pi_check):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['builtin']
    modules = ['Spark']
    spark_prefix = '$EBROOTSPARK'
    executor_memory = '15g'
    maintainers = ['TM', 'RS']
    tags |= {'production'}

    @run_before('run')
    def set_num_workers_and_cores(self):
        self.num_workers = self.current_partition.processor.num_cores
        self.exec_cores = self.num_workers // 4
