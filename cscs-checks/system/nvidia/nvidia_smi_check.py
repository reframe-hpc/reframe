# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


@rfm.simple_test
class nvidia_smi_check(rfm.RunOnlyRegressionTest):
    gpu_mode = parameter(['accounting', 'compute', 'ecc'])
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['builtin']
    executable = 'nvidia-smi'
    executable_opts = ['-a', '-d']
    num_tasks = 1
    num_tasks_per_node = 1
    exclusive = True
    tags = {'maintenance', 'production'}
    maintainers = ['VH']
    mode_values = variable(typ.Dict[str, str], value={
        'accounting': 'Enabled',
        'compute': 'Exclusive_Process',
        'ecc': 'Enabled'
    })

    @run_before('run')
    def set_display_opt(self):
        self.executable_opts.append(self.gpu_mode.upper())

    @run_before('sanity')
    def set_sanity(self):
        modeval = self.mode_values[self.gpu_mode]
        if self.gpu_mode == 'ecc':
            patt = rf'Current\s+: {modeval}'
        else:
            patt = rf'{self.gpu_mode.capitalize()} Mode\s+: {modeval}'

        num_gpus_detected = sn.count(sn.findall(patt, self.stdout))
        num_gpus_all = self.num_tasks

        # We can't an use f-string here, because it will misinterpret the
        # placeholders for the sanity function message
        errmsg = ('{0} out of {1} GPU(s) have the correct %s mode' %
                  self.gpu_mode)
        self.sanity_patterns = sn.assert_eq(
            num_gpus_detected, num_gpus_all, errmsg
        )
