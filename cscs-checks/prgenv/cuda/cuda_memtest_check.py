# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class cuda_memtest_check(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'dom:gpu', 'ault:amdv100',
                     'ault:intelv100']
    valid_prog_environs = ['PrgEnv-cray']
    descr = 'Flexible CUDA Memtest'
    maintainers = ['TM', 'SK']
    num_tasks_per_node = 1
    num_tasks = 0
    num_gpus_per_node = 1
    modules = ['cudatoolkit']
    src_url = ('https://downloads.sourceforge.net/project/cudagpumemtest/'
               'cuda_memtest-1.2.3.tar.gz')
    prebuild_cmds = [
        'wget %s' % src_url,
        'tar -xzf cuda_memtest-1.2.3.tar.gz',
        'cd cuda_memtest-1.2.3',
        'patch -p1 < ../cuda_memtest-1.2.3.patch'
    ]
    build_system = 'Make'
    executable = './cuda_memtest-1.2.3/cuda_memtest'
    executable_opts = ['--disable_test', '6', '--num_passes', '1']
    tags = {'diagnostic', 'ops', 'craype', 'health'}

    @run_before('sanity')
    def set_sanity_patterns(self):
        valid_test_ids = {i for i in range(11) if i not in {6, 9}}
        assert_finished_tests = [
            sn.assert_eq(
                sn.count(sn.findall('Test%s finished' % test_id, self.stdout)),
                self.job.num_tasks
            )
            for test_id in valid_test_ids
        ]
        self.sanity_patterns = sn.all([
            *assert_finished_tests,
            sn.assert_not_found('(?i)ERROR', self.stdout),
            sn.assert_not_found('(?i)ERROR', self.stderr)])
