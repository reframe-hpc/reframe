# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from library.microbenchmarks.gpu.gpu_burn import GpuBurn

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
    perf_floor = variable(float, value=-1.0)

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = self.gpu_usage_sanity()

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
        return sn.evaluate(sn.all([
            sn.assert_ge(sn.count(self.nodes_reported), 1),
            set(self.nodes_reported).issubset(full_node_set),
            sn.all(
                map(lambda x, y: self.burn_time/x <= y, time_reported, usage)
            ),
            sn.all(
                map(lambda x: x >= self.burn_time, time_reported)
            )
        ]))

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
            'nodes_reported':sn.count(self.nodes_reported)
        }


