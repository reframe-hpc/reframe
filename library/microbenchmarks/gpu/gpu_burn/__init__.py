# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['GpuBurnBase']


class GpuBurnBase(rfm.RegressionTest, pin_prefix=True):
    # GPU build variables
    gpu_build = variable(str)
    gpu_arch = variable(str, type(None), value=None)

    executable_opts = required
    num_tasks = required

    def __init__(self):
        self.descr = 'GPU burn test'
        self.build_system = 'Make'
        self.executable = './gpu_burn.x'
        self.num_tasks_per_node = 1
        self.sanity_patterns = self.assert_num_tasks()
        patt = (r'^\s*\[[^\]]*\]\s*GPU\s+\d+\(\S*\):\s+(?P<perf>\S*)\s+GF\/s'
                r'\s+(?P<temp>\S*)\s+Celsius')
        self.perf_patterns = {
            'perf': sn.min(sn.extractall(patt, self.stdout, 'perf', float)),
            'temp': sn.max(sn.extractall(patt, self.stdout, 'temp', float)),
        }

    @rfm.run_before('compile')
    def set_gpu_build(self):
        if self.gpu_build == 'cuda':
            self.build_system.makefile = 'makefile.cuda'
            if self.gpu_arch:
                self.build_system.cxxflags = [f'-arch=compute_{self.gpu_arch}',
                                              f'-code=sm_{self.gpu_arch}']
        elif self.gpu_build == 'hip':
            self.build_system.makefile = 'makefile.hip'
            if self.gpu_arch:
                self.build_system.cxxflags = [
                    f'--amdgpu-target={self.gpu_arch}'
                ]
        else:
            raise ValueError('unknown gpu_build option')

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks * self.num_gpus_per_node

    @sn.sanity_function
    def assert_num_tasks(self):
        return sn.assert_eq(sn.count(sn.findall(
            r'^\s*\[[^\]]*\]\s*GPU\s*\d+\(OK\)', self.stdout)
        ), self.num_tasks_assigned)

    @rfm.run_before('performance')
    def report_nid_with_smallest_flops(self):
        regex = r'\[(\S+)\] GPU\s+\d\(OK\): (\d+) GF/s'
        rptf = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        self.nids = sn.extractall(regex, rptf, 1)
        self.flops = sn.extractall(regex, rptf, 2, float)

        # Find index of smallest flops and update reference dictionary to
        # include our patched units
        index = self.flops.evaluate().index(min(self.flops))
        unit = f'GF/s ({self.nids[index]})'
        for key, ref in self.reference.items():
            if not key.endswith(':temp'):
                self.reference[key] = (*ref[:3], unit)
