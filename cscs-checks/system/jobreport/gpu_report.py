# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import time

from reframe.core.exceptions import SanityError
from hpctestlib.microbenchmarks.gpu.gpu_burn import gpu_burn_check


@rfm.simple_test
class gpu_usage_report_check(gpu_burn_check):
    '''Check the output from the job report.

    This check uses the gpu burn app and checks that the job report produces
    senssible gpu usage data. The total number of reported nodes is used as
    a performance metric, given that the number of reported nodes might be
    lower than the total number of nodes. The lower threshold of the performace
    check can be adjusted with the test variable `perf_floor`.
    '''

    valid_systems = ['daint:gpu', 'dom:gpu']
    descr = 'Check GPU usage from job report'
    gpu_build = 'cuda'
    num_tasks = 2
    num_gpus_per_node = 1
    perf_floor = variable(float, value=-0.2)
    tags = {'production'}

    @run_before('run')
    def set_launcher_opts(self):
        '''Make slurm's output unbuffered.

        Without this, the jobreport data gets written into the stdout after the
        job is completed, causing the sanity function to issue a sanity error.
        '''
        self.job.launcher.options = ['-u']

    @sanity_function
    def assert_jobreport_success(self):
        '''Extend sanity and wait for the jobreport.

        If a large number of nodes is used, the final jobreport output happens
        much later after job has already completed (this could be up to 25s).
        However, this wait might not be necessary for a small node count, so
        we use the try/except block below to wait only if a first call to the
        sanity function does not succeed.
        '''

        try:
            sn.evaluate(self.gpu_usage_sanity())
        except SanityError:
            time.sleep(25)

        return self.assert_successful_burn_count(), self.gpu_usage_sanity()

    @deferrable
    def gpu_usage_sanity(self):
        '''Verify that the jobreport output has sensible numbers.

        The GPU usage is verified by assuming that in the worst case scenario,
        the usage is near 100% during the burn, and 0% outside the burn period.
        Lastly, the GPU usage time for each node is also asserted to be greater
        or equal than the burn time.
        '''

        # Parse job report data
        patt = r'^\s*(\w*)\s*(\d+)\s*%\s*\d+\s*MiB\s*\d+:\d+:(\d+)'
        self.nodes_reported = sn.extractall(patt, self.stdout, 1)
        self.num_tasks_assigned = self.num_tasks * self.num_gpus_per_node
        usage = sn.extractall(patt, self.stdout, 2, int)
        time_reported = sn.extractall(patt, self.stdout, 3, int)
        return sn.all([
            sn.assert_ge(sn.count(self.nodes_reported), 1),
            sn.all(
                map(lambda x, y: self.duration/x <= y/100, time_reported, usage)
            ),
            sn.assert_ge(sn.min(time_reported), self.duration)
        ])

    @deferrable
    def assert_successful_burn_count(self):
        '''Assert that the expected successful burn count is reported.'''
        return sn.assert_eq(sn.count(sn.findall(r'^GPU\s*\d+\(OK\)',
                                                self.stdout)),
                            self.num_tasks_assigned)
