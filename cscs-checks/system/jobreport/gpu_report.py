# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import time

from reframe.core.exceptions import SanityError
from hpctestlib.microbenchmarks.gpu.gpu_burn import GpuBurn


@rfm.simple_test
class gpu_usage_report_check(GpuBurn):
    '''Check the output from the job report.

    This check uses the gpu burn app and checks that the job report produces
    senssible gpu usage data. The total number of reported nodes is used as
    a performance metric, given that the number of reported nodes might be
    lower than the total number of nodes. The lower threshold of the performace
    check can be adjusted with the test variable `perf_floor`.
    '''

    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    descr = 'Check GPU usage from job report'
    gpu_build = 'cuda'
    modules = ['craype-accel-nvidia60', 'cdt-cuda']
    num_tasks = 0
    num_gpus_per_node = 1
    burn_time = variable(int, value=10)
    executable_opts = ['-d', f'{burn_time}']
    perf_floor = variable(float, value=-0.2)
    tags = {'production'}

    @rfm.run_before('run')
    def set_launcher_opts(self):
        '''Make slurm's output unbuffered.

        Without this, the jobreport data gets written into the stdout after the
        job is completed, causing the sanity function to issue a sanity error.
        '''
        self.job.launcher.options = ['-u']

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        '''Set sanity patterns and wait for the jobreport.

        If a large number of nodes is used, the final jobreport output happens
        much later after job has already completed (this could be up to 25s).
        However, this wait might not be necessary for a small node count, so
        we use the try/except block below to wait only if a first call to the
        sanity function does not succeed.
        '''

        super().set_sanity_patterns()
        self.sanity_patterns = sn.all([
            self.sanity_patterns, self.gpu_usage_sanity()
        ])
        try:
            sn.evaluate(self.gpu_usage_sanity())
        except SanityError:
            time.sleep(25)

    @sn.sanity_function
    def gpu_usage_sanity(self):
        '''Verify that the jobreport output has sensible numbers.

        This function asserts that the nodes reported are at least a subset of
        all nodes used by the gpu burn app. Also, the GPU usage is verified by
        assuming that in the worst case scenario, the usage is near 100% during
        the burn, and 0% outside the burn period. Lastly, the GPU usage time
        for each node is also asserted to be greater or equal than the burn
        time.
        '''

        # Get set with all nodes
        patt = r'^\s*\[([^\]]*)\]\s*GPU\s*\d+\(OK\)'
        full_node_set = set(sn.extractall(patt, self.stdout, 1))

        # Parse job report data
        patt = r'^\s*(\w*)\s*(\d+)\s*%\s*\d+\s*MiB\s*\d+:\d+:(\d+)'
        self.nodes_reported = sn.extractall(patt, self.stdout, 1)
        usage = sn.extractall(patt, self.stdout, 2, int)
        time_reported = sn.extractall(patt, self.stdout, 3, int)
        return sn.all([
            sn.assert_ge(sn.count(self.nodes_reported), 1),
            set(self.nodes_reported).issubset(full_node_set),
            sn.all(
                map(lambda x, y: self.burn_time/x <= y, time_reported, usage)
            ),
            sn.assert_ge(sn.min(time_reported), self.burn_time)
        ])

    @rfm.run_before('performance')
    def set_perf_patterns(self):
        '''The number of reported nodes can be used as a perf metric.

        For now, the low limit can go to zero, but this can be set to a more
        restrictive value.
        '''

        self.reference = {
            '*': {
                'nodes_reported': (self.num_tasks, self.perf_floor, 0, 'nodes')
            },
        }
        self.perf_patterns = {
            'nodes_reported': sn.count(self.nodes_reported)
        }
