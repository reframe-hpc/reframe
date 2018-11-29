import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.launchers.registry import getlauncher

@rfm.simple_test
class SpecAccelCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'SPEC-accel benchmark'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['cudatoolkit/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52']

        #self.sourcesdir needed for cscs-default config file
        app_source = os.path.join(self.current_system.resourcesdir,
                                  'SPEC_ACCELv1.2')
        self.prebuild_cmd = ['cp -r %s/* .' % app_source,
                             './install.sh -d . -f']

        # I just want prebuild_cmd, but no action for the build_system
        # is not supported, so I find it something useless to do
        self.build_system = 'SingleSource'
        self.sourcepath = './benchspec/ACCEL/353.clvrleaf/src/timer_c.c'
        self.build_system.cflags = ['-c']

        self.pre_run = ['source ./shrc', 'mv cscs-default config']

        benchmarks = ['systest', 'tpacf', 'stencil', 'lbm', 'fft', 'spmv',
                      'mriq', 'bfs', 'cutcp', 'kmeans', 'lavamd', 'cfd', 'nw',
                      'hotspot', 'lud', 'ge', 'srad', 'heartwall', 'bplustree']

        self.executable = 'runspec'
        self.executable_opts = ['--config=cscs-default',
                                '--platform NVIDIA',
                                '--tune=base',
                                '--device GPU'] + benchmarks

        outfile = sn.getitem(sn.glob('result/ACCEL.*.log'), 0)
        self.sanity_patterns = sn.all([sn.assert_found(
                                        r'Success.*%s' % bn, outfile)
                                          for bn in benchmarks])

        refs = { bench_name : (runtime, -0.1, None)
                    for (bench_name, runtime) in
                        zip(benchmarks,
                            [10.7, 13.5, 17.0, 10.9, 11.91, 27.8, 7.0,
                             23.1, 10.8, 55.9, 8.7, 24.4, 16.2,
                             15.7, 15.6, 11.1, 20.0, 41.9, 26.2]
                        )
                }

        self.reference = {
            'dom:gpu' : refs,
            'daint:gpu' : refs
        }

        self.perf_patterns = {
            bench_name: self.extract_average(outfile, bench_name)
                for bench_name in benchmarks
        }

        self.maintainers = ['SK']
        self.tags = {'benchmark'}

    def setup(self, partition, environ, **job_opts):
        self.num_tasks = 1
        self.num_tasks_per_node = 1

        super().setup(partition, environ, **job_opts)
        # The job launcher has to be changed since the `runspec`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()

    @sn.sanity_function
    def extract_average(self, ofile, bench_name):
        runs = sn.extractall(r'Success.*%s.*runtime=(?P<rt>[0-9.]+)'
                                % bench_name, ofile, 'rt', float)
        return sum(runs)/sn.count(runs)
