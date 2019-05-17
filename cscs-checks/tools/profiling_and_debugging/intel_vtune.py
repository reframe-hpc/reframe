import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['C++'], ['F90'])
class IntelVTuneAmplifierTest(rfm.RegressionTest):
    '''This test checks Intel VTune Amplifier:
    https://software.intel.com/en-us/intel-vtune-amplifier-xe
    and the -Cperf slurm constraint (that sets perf_event_paranoid to 0 for
    advanced performance analysis)
    '''
    def __init__(self, lang):
        super().__init__()
        self.name = 'Intel_VTuneAmplifier_%s' % lang.replace('+', 'p')
        self.descr = self.name
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-cray': ['-O2', '-g', '-homp'],
            'PrgEnv-intel': ['-O2', '-g', '-qopenmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp']
        }
        self.sourcesdir = os.path.join('src', lang)
        self.executable = 'amplxe-cl'
        self.executable_opts = ['-trace-mpi -collect hotspots -r ./hotspots',
                                '-data-limit=0 ./jacobi']
        self.build_system = 'Make'
        if lang == 'F90':
            self.build_system.max_concurrency = 1

        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_iterations = 10
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.version_rpt = 'Intel_VTuneAmplifier_version.rpt'
        self.summary_rpt = 'Intel_VTuneAmplifier_summary.rpt'
        self.paranoid_rpt = 'Intel_VTuneAmplifier_paranoid.rpt'
        self.post_run = [
            'amplxe-cl -V &> %s' % self.version_rpt,
            'amplxe-cl -R hotspots -r hotspots* -column="CPU Time:Self" &>%s' %
            self.summary_rpt,
            'srun -n1 cat /proc/sys/kernel/perf_event_paranoid &> %s' %
            self.paranoid_rpt,
        ]
        self.maintainers = ['JG']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        partitiontype = partition.fullname.split(':')[1]
        if partitiontype == 'gpu':
            self.job.options = ['--constraint="gpu&perf"']
        elif partitiontype == 'mc':
            self.job.options = ['--constraint="mc&perf"']

        if self.current_system.name == 'dom':
            vtuneversion = '2019'
            toolsversion = '579888'
        elif self.current_system.name == 'daint':
            vtuneversion = '2018'
            toolsversion = '551022'

        self.pre_run = [
            'source $INTEL_PATH/../vtune_amplifier_%s/amplxe-vars.sh' %
            vtuneversion,
            'amplxe-cl -help collect |tail -20',
        ]
        self.sanity_patterns = sn.all([
            # check the job:
            sn.assert_found('SUCCESS', self.stdout),
            sn.assert_found(r'amplxe: Executing actions \d+ %', self.stderr),
            # check the version:
            sn.assert_eq(sn.extractsingle(
                r'I*.\(build\s(?P<toolsversion>\d+)\s*.', self.version_rpt,
                'toolsversion'), toolsversion),
            # check the perf_event setting:
            sn.assert_eq(sn.extractsingle(r'(?P<perfevent>\d)',
                         self.paranoid_rpt, 'perfevent'), '0'),
            # check the hotspots:
            sn.assert_found(r'^[jJ]acobi.*\$omp\$parallel@\d+\s+\d+.\d+s',
                            self.summary_rpt),
        ])
