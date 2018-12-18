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
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray']
        self.modules = ['craype-accel-nvidia60']

        self.configs = {
            'PrgEnv-gnu':  'cscs-gnu',
            'PrgEnv-cray': 'cscs-cray',
        }

        #self.sourcesdir needed for cscs-* config files
        app_source = os.path.join(self.current_system.resourcesdir,
                                  'SPEC_ACCELv1.2')
        self.prebuild_cmd = ['cp -r %s/* .' % app_source,
                             './install.sh -d . -f']

        # I just want prebuild_cmd, but no action for the build_system
        # is not supported, so I find it something useless to do
        self.build_system = 'SingleSource'
        self.sourcepath = './benchspec/ACCEL/353.clvrleaf/src/timer_c.c'
        self.build_system.cflags = ['-c']

        #self.benchmarks = ['systest', 'tpacf', 'stencil', 'lbm', 'fft', 'spmv',
        #                   'mriq', 'bfs', 'cutcp', 'kmeans', 'lavamd', 'cfd', 'nw',
        #                   'hotspot', 'lud', 'ge', 'srad', 'heartwall', 'bplustree']

        #self.runtimes = {
        #    'PrgEnv-gnu':  [10.7, 13.5, 17.0, 10.9, 11.91, 27.8,
        #                    7.0, 23.1, 10.8, 38.4, 8.7, 24.4, 16.2,
        #                    15.7, 15.6, 11.1, 20.0, 41.9, 26.2],
        #    'PrgEnv-cray': [10.7, 13.5, 17.0, 10.9, 11.91, 27.8,
        #                    7.0, 23.1, 10.8, 24.9, 8.7, 24.4, 16.2,
        #                    15.7, 15.6, 11.1, 20.0, 41.9, 26.2],
        #}
        self.benchmarks = ['ostencil', 'olbm', 'omriq', 'md', 'ep', 'clvrleaf', 'cg',
                           'seismic', 'sp', 'csp', 'miniGhost', 'ilbdc', 'swim', 'bt']

        self.runtimes = {
            'PrgEnv-gnu':  len(self.benchmarks) * [100],
            'PrgEnv-cray': len(self.benchmarks) * [100],
        }

        self.refs = {
            env: { bench_name : (rt, None, 0.1)
                     for (bench_name, rt) in
                       zip(self.benchmarks, self.runtimes[env])
                 }
            for env in self.valid_prog_environs   
        }

        self.num_tasks = 12
        self.num_tasks_per_node = 12
        self.time_limit = (0, 30, 0)

        self.executable = 'runspec'

        outfile = sn.getitem(sn.glob('result/ACCEL.*.log'), 0)
        self.sanity_patterns = sn.all([sn.assert_found(
                                       r'Success.*%s' % bn, outfile)
                                       for bn in self.benchmarks
                                      ])

        self.perf_patterns = {
            bench_name: self.extract_average(outfile, bench_name)
                for bench_name in self.benchmarks
        }

        self.maintainers = ['SK']
        self.tags = {'diagnostic'}

    def setup(self, partition, environ, **job_opts):

        self.pre_run = ['source ./shrc', 'mv %s config' % self.configs[environ.name]]
        self.executable_opts = ['--config=%s' % self.configs[environ.name],
                                '--platform NVIDIA',
                                '--tune=base',
                                '--device GPU'] + self.benchmarks
        self.reference = {
            'dom:gpu':   self.refs[environ.name],
            'daint:gpu': self.refs[environ.name]
        }

        super().setup(partition, environ, **job_opts)
        # The job launcher has to be changed since the `runspec`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()

    @sn.sanity_function
    def extract_average(self, ofile, bench_name):
        runs = sn.extractall(r'Success.*%s.*runtime=(?P<rt>[0-9.]+)'
                                % bench_name, ofile, 'rt', float)
        return sum(runs)/sn.count(runs)
