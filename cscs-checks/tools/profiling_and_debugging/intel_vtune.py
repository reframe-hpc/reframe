# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['C++'], ['F90'])
class IntelVTuneAmplifierTest(rfm.RegressionTest):
    '''This test checks Intel VTune Amplifier and the -Cperf slurm constraint
    (that sets perf_event_paranoid to 0 for advanced performance analysis)
    https://software.intel.com/en-us/intel-vtune-amplifier-xe
    '''
    def __init__(self, lang):
        super().__init__()
        self.name = 'Intel_VTuneAmplifier_%s' % lang.replace('+', 'p')
        self.descr = self.name
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['vtune_amplifier']
        self.sourcesdir = os.path.join('src', lang)
        self.build_system = 'Make'
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.executable = 'amplxe-cl'
        self.target_executable = './jacobi'
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-g', '-O2', '-fopenmp'],
            'PrgEnv-cray': ['-g', '-O2', '-homp'],
            'PrgEnv-intel': ['-g', '-O2', '-qopenmp'],
            'PrgEnv-pgi': ['-g', '-O2', '-mp']
        }
        self.executable_opts = ['-trace-mpi -collect hotspots -r ./hotspots',
                                '-data-limit=0 %s' % self.target_executable]
        self.exclusive = True
        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        num_iterations = 10
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.version_rpt = 'version.rpt'
        self.paranoid_rpt = 'paranoid.rpt'
        self.summary_rpt = 'summary.rpt'
        self.pre_run = [
            'mv %s %s' % (self.executable, self.target_executable),
            '%s --version &> %s' % (self.executable, self.version_rpt),
        ]
        self.post_run = ['%s --help | head -20' % self.executable]
        self.maintainers = ['JG']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        run_cmd = ' '.join(self.job.launcher.command(self.job))
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.post_run += [
            '%s -R hotspots -r hotspots* -column="CPU Time:Self" &> %s' %
            (self.executable, self.summary_rpt),
            '%s cat /proc/sys/kernel/perf_event_paranoid &> %s' %
            (run_cmd, self.paranoid_rpt),
        ]
        partitiontype = partition.fullname.split(':')[1]
        if partitiontype == 'gpu':
            self.job.options = ['--constraint="gpu&perf"']
        elif partitiontype == 'mc':
            self.job.options = ['--constraint="mc&perf"']

        system_default_toolversion = {
            'daint': '597835',  # 2019 Update 4
            'dom': '597835',    # 2019 Update 4
        }
        toolsversion = system_default_toolversion[self.current_system.name]
        self.sanity_patterns = sn.all([
            # check the job:
            sn.assert_found('SUCCESS', self.stdout),
            sn.assert_found(r'amplxe: Executing actions \d+ %', self.stderr),
            # check the tool's version:
            sn.assert_eq(sn.extractsingle(
                r'I*.\(build\s(?P<toolsversion>\d+)\s*.', self.version_rpt,
                'toolsversion'), toolsversion),
            # check the perf_event setting:
            sn.assert_eq(sn.extractsingle(r'(?P<perfevent>\d)',
                         self.paranoid_rpt, 'perfevent'), '0'),
            # check the hotspots:
            sn.assert_found(r'^.*\$omp\$parallel.*@(?P<line>\d+)\s+'
                            r'(?P<sec>\S+)s',
                            self.summary_rpt),
        ])
