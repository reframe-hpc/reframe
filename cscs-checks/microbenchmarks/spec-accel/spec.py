# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.launchers.registry import getlauncher


class SpecAccelCheckBase(rfm.RegressionTest):
    def __init__(self, prg_envs):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = prg_envs
        self.modules = ['craype-accel-nvidia60']

        self.configs = {
            'PrgEnv-gnu':  'cscs-gnu',
            'PrgEnv-cray': 'cscs-cray',
            'PrgEnv-pgi': 'cscs-pgi',
        }

        app_source = os.path.join(self.current_system.resourcesdir,
                                  'SPEC_ACCELv1.2')
        self.prebuild_cmd = ['cp -r %s/* .' % app_source,
                             './install.sh -d . -f']

        # I just want prebuild_cmd, but no action for the build_system
        # is not supported, so I find it something useless to do
        self.build_system = 'SingleSource'
        self.sourcepath = './benchspec/ACCEL/353.clvrleaf/src/timer_c.c'
        self.build_system.cflags = ['-c']

        self.refs = {
            env: {bench_name: (rt, None, 0.1, 'Seconds')
                  for (bench_name, rt) in
                  zip(self.benchmarks[env], self.exec_times[env])}
            for env in self.valid_prog_environs
        }

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.time_limit = (0, 30, 0)

        self.executable = 'runspec'

        outfile = sn.getitem(sn.glob('result/ACCEL.*.log'), 0)
        self.sanity_patterns_ = {
            env: sn.all([sn.assert_found(
                r'Success.*%s' % bn, outfile) for bn in self.benchmarks[env]])
            for env in self.valid_prog_environs
        }

        self.perf_patterns_ = {
            env: {bench_name: sn.avg(sn.extractall(
                  r'Success.*%s.*runtime=(?P<rt>[0-9.]+)' % bench_name,
                  outfile, 'rt', float))
                  for bench_name in self.benchmarks[env]}
            for env in self.valid_prog_environs
        }

        self.maintainers = ['SK']
        self.tags = {'diagnostic', 'external-resources'}

    def setup(self, partition, environ, **job_opts):
        self.pre_run = ['source ./shrc', 'mv %s config' %
                        self.configs[environ.name]]
        self.executable_opts = ['--config=%s' % self.configs[environ.name],
                                '--platform NVIDIA',
                                '--tune=base',
                                '--device GPU'] + self.benchmarks[environ.name]
        self.reference = {
            'dom:gpu':   self.refs[environ.name],
            'daint:gpu': self.refs[environ.name]
        }

        self.sanity_patterns = self.sanity_patterns_[environ.name]
        self.perf_patterns = self.perf_patterns_[environ.name]

        super().setup(partition, environ, **job_opts)
        # The job launcher has to be changed since the `runspec`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class SpecAccelCheckOpenCL(SpecAccelCheckBase):
    def __init__(self):
        self.descr = 'SPEC-accel benchmark OpenCL'
        valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-pgi']

        self.benchmarks = {
            'PrgEnv-gnu': ['systest', 'tpacf', 'stencil', 'lbm', 'fft',
                           'spmv', 'mriq', 'bfs', 'cutcp', 'kmeans',
                           'lavamd', 'cfd', 'nw', 'hotspot', 'lud',
                           'ge', 'srad', 'heartwall', 'bplustree'],
            'PrgEnv-cray': ['systest', 'tpacf', 'stencil', 'lbm', 'fft',
                            'spmv', 'mriq', 'bfs', 'cutcp', 'kmeans',
                            'lavamd', 'cfd', 'nw', 'hotspot', 'lud',
                            'ge', 'srad', 'heartwall', 'bplustree'],
            'PrgEnv-pgi': ['systest', 'tpacf', 'stencil', 'lbm', 'fft',
                           'spmv', 'mriq', 'bfs', 'kmeans',
                           'lavamd', 'cfd', 'nw', 'hotspot', 'lud',
                           'ge', 'srad', 'heartwall', 'bplustree'],
        }

        self.exec_times = {
            'PrgEnv-gnu':  [10.7, 13.5, 17.0, 10.9, 11.91, 27.8,
                            7.0, 23.1, 10.8, 38.4, 8.7, 24.4, 16.2,
                            15.7, 15.6, 11.1, 20.0, 41.9, 26.2],
            'PrgEnv-cray': [10.7, 13.5, 17.0, 10.9, 11.91, 27.8,
                            7.0, 23.1, 10.8, 24.9, 8.7, 24.4, 16.2,
                            15.7, 15.6, 11.1, 20.0, 41.9, 26.2],
            'PrgEnv-pgi': [10.7, 30, 17.0, 10.9, 11.91, 27.8,
                           7.0, 23.1, 24.9, 8.7, 24.4, 16.2,
                           15.7, 15.6, 11.1, 20.0, 41.9, 26.2],
        }

        super().__init__(valid_prog_environs)


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class SpecAccelCheckOpenACC(SpecAccelCheckBase):
    def __init__(self):
        self.descr = 'SPEC-accel benchmark OpenACC'
        valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']

        self.benchmarks = {
            'PrgEnv-cray': ['ostencil', 'olbm', 'omriq', 'md', 'ep',
                            'clvrleaf', 'cg', 'seismic', 'sp', 'csp',
                            'miniGhost', 'ilbdc', 'swim', 'bt'],
            'PrgEnv-pgi': ['ostencil', 'olbm', 'omriq', 'md', 'ep',
                           'clvrleaf', 'cg', 'seismic', 'sp', 'csp',
                           'ilbdc', 'swim', 'bt'],
        }

        self.exec_times = {
            'PrgEnv-cray': [18, 26, 121, 20, 73, 59, 41,
                            50, 71, 34, 72, 41, 34, 378],
            'PrgEnv-pgi': [18, 40, 116, 20, 71, 57, 45,
                           56, 33, 32, 41, 48, 15]
        }

        super().__init__(valid_prog_environs)
