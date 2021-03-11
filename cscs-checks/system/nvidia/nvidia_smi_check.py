# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


def sanity_evaluation(regex, regex_group, output, refvalue, field_msg, num_nodes, nodelist):
    all_tested_nodes = sn.evaluate(sn.extractall(regex, output, regex_group))
    num_tested_nodes = len(all_tested_nodes)
    node_failure_msg = ('Requested %s node(s), but found %s node(s)' %
                   (num_nodes, num_tested_nodes))
    sn.evaluate(sn.assert_eq(num_tested_nodes, num_nodes,
                                msg=node_failure_msg))

    failures = []
    for i in range(len(all_tested_nodes)):
        stats = all_tested_nodes[i]
        node = nodelist[i]
        if stats != refvalue:
            failures.append(node)

    the_failure_msg = '%s is not set to "%s" for node(s): %s' % (field_msg, refvalue, ','.join(failures))
    sn.evaluate(sn.assert_eq(len(failures), 0, msg=the_failure_msg))


class NVIDIASMIBaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for NVIDIA SMI tests'''

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['builtin']
        self.executable = 'nvidia-smi'
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.exclusive = True
        self.tags = {'nvidia', 'maintenance', 'production', 'single-node'}
        self.maintainers = ['VH']


@rfm.simple_test
class NVIDIASMIAccountCheck(NVIDIASMIBaseCheck):
    '''Check whether Accounting mode is enabled'''

    def __init__(self):
        super().__init__()
        self.executable_opts = ['-a', '-d', 'ACCOUNTING']
        self.sanity_patterns = self.eval_sanity()

    @sn.sanity_function
    def eval_sanity(self):
        sanity_evaluation(
            regex=r'Accounting Mode\s+: (?P<stats>\S+)',
            regex_group='stats',
            output=self.stdout,
            refvalue='Enabled',
            field_msg='Accounting mode',
            num_nodes=self.job.num_tasks,
            nodelist=self.job.nodelist,
        )
        return True


@rfm.simple_test
class NVIDIASMIComputeModeCheck(NVIDIASMIBaseCheck):
    '''Check whether Compute mode is set to Exclusive_Process'''

    def __init__(self):
        super().__init__()
        self.executable_opts = ['-a', '-d', 'COMPUTE']
        self.sanity_patterns = self.eval_sanity()

    @sn.sanity_function
    def eval_sanity(self):
        sanity_evaluation(
            regex=r'Compute Mode\s+: (?P<stats>\S+)',
            regex_group='stats',
            output=self.stdout,
            refvalue='Exclusive_Process',
            field_msg='Compute mode',
            num_nodes=self.job.num_tasks,
            nodelist=self.job.nodelist,
        )
        return True


@rfm.simple_test
class NVIDIASMIECCModeCheck(NVIDIASMIBaseCheck):
    '''Check whether ECC mode is enabled'''

    def __init__(self):
        super().__init__()
        self.executable_opts = ['-a', '-d', 'ECC']
        self.sanity_patterns = self.eval_sanity()

    @sn.sanity_function
    def eval_sanity(self):
        sanity_evaluation(
            regex=r'Current\s+: (?P<stats>\S+)',
            regex_group='stats',
            output=self.stdout,
            refvalue='Enabled',
            field_msg='Current ECC mode',
            num_nodes=self.job.num_tasks,
            nodelist=self.job.nodelist,
        )
        sanity_evaluation(
            regex=r'Pending\s+: (?P<stats>\S+)',
            regex_group='stats',
            output=self.stdout,
            refvalue='Enabled',
            field_msg='Pending ECC mode',
            num_nodes=self.job.num_tasks,
            nodelist=self.job.nodelist,
        )
        return True
