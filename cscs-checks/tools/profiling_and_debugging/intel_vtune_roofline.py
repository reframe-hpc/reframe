import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*[[repeat, toolsversion, datalayout]
                          for repeat in ['500000']
                          for toolsversion in ['597835']
                          for datalayout in ['G3_AOS_SCALAR', 'G3_SOA_SCALAR',
                                             'G3_AOS_VECTOR', 'G3_SOA_VECTOR']
                          ])
class IntelRooflineVtuneTest(rfm.RegressionTest):
    '''This test checks the values reported by Vtune for roofline modeling:
       https://docs.nersc.gov/programming/performance-debugging-tools/roofline/

    Example result on 1 core of Intel Broadwell CPU (E5-2695 v4):
        G3_AOS_SCALAR: DP GFLOPS:  3.162 Time: 0.854s <-- slow
        G3_AOS_VECTOR: DP GFLOPS:  5.731 Time: 0.440s
        G3_SOA_SCALAR: DP GFLOPS:  3.183 Time: 0.848s
        G3_SOA_VECTOR: DP GFLOPS: 21.423 Time: 0.134s <-- fast
    '''
    def __init__(self, repeat, toolsversion, datalayout):
        super().__init__()
        self.descr = 'Roofline Analysis test with Intel Vtune'
        self.debug = False
        self.valid_systems = ['dom:mc']
        # Reporting MFLOPS is not available on Intel Haswell cpus, see
        # https://www.intel.fr/content/dam/www/public/us/en/documents/manuals/
        # 64-ia-32-architectures-software-developer-vol-1-manual.pdf
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['vtune_amplifier']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'intel_advisor')
        self.build_system = 'SingleSource'
        self.sourcepath = '_roofline.cpp'
        self.executable = 'amplxe-cl'
        self.target_executable = './roof.exe'
        self.build_system.cppflags = ['-D_ADVISOR',
                                      '-I$VTUNE_AMPLIFIER_2019_DIR/include']
        self.prgenv_flags = {
            'PrgEnv-intel': ['-g', '-O2', '-std=c++11', '-restrict'],
            # TODO: evaluate '-qopt-streaming-stores', 'always',
        }
        self.build_system.ldflags = ['-L$VTUNE_AMPLIFIER_2019_DIR/lib64',
                                     '-littnotify']
        self.roofline_rpt = '%s.rpt' % self.target_executable
        self.version_rpt = 'version.rpt'
        self.roofline_ref = 'reference.rpt'
        self.prebuild_cmd = [
            'patch -s < ADVISOR/roofline_template.patch',
            'sed -e "s-XXXX-%s-" -e "s-YYYY-%s-" %s &> %s' %
            (repeat, datalayout, 'roofline_template.cpp', '_roofline.cpp')
        ]
        self.exclusive = True
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.pre_run = [
            'mv %s %s' % (self.executable, self.target_executable),
            '%s --version &> %s' % (self.executable, self.version_rpt),
            '%s -help | head -20' % self.executable,
        ]
        self.roofdir = './roof.dir'
        self.executable_opts = [
            '-start-paused -r %s -collect hpc-performance -data-limit=0 '
            '--search-dir src:rp=. --trace-mpi -- %s' %
            (self.roofdir, self.target_executable)
        ]
        # NOTE: -allow-multiple-runs requires to install vtune drivers
        # TODO: -collect memory-access
        self.maintainers = ['JG']
        self.tags = {'scs', 'external-resources'}
        self.sanity_patterns = sn.all([
            sn.assert_found('loop complete.', self.stdout),
            sn.assert_eq(sn.extractsingle(
                r'I*.\(build\s(?P<toolsversion>\d+)\s*.',
                self.version_rpt, 'toolsversion'), toolsversion),
        ])
        # References for Intel Broadwell CPU (E5-2695 v4):
        references = {
            'G3_AOS_SCALAR': {
                'dom:mc': {
                    'gflops': (3.1, -0.1, None, 'Gflop/s'),
                    'compare_sec': (0, -0.1, 0.1, 'seconds'),
                    'compare_gflops': (0, -0.2, 0.2, 'Gflop/s'),
                }
            },
            'G3_AOS_VECTOR': {
                'dom:mc': {
                    'gflops': (5.7, -0.1, None, 'Gflop/s'),
                    'compare_sec': (0, -0.1, 0.1, 'seconds'),
                    'compare_gflops': (0, -0.2, 0.2, 'Gflop/s'),
                }
            },
            'G3_SOA_SCALAR': {
                'dom:mc': {
                    'gflops': (3.1, -0.1, None, 'Gflop/s'),
                    'compare_sec': (0, -0.1, 0.1, 'seconds'),
                    'compare_gflops': (0, -0.2, 0.2, 'Gflop/s'),
                }
            },
            'G3_SOA_VECTOR': {
                'dom:mc': {
                    'gflops': (21.0, -0.1, None, 'Gflop/s'),
                    'compare_sec': (0, -0.1, 0.1, 'seconds'),
                    'compare_gflops': (0, -0.2, 0.2, 'Gflop/s'),
                }
            },
        }
        self.reference = references[datalayout]
        self.perf_patterns = {
            'gflops': self.gflops_reported,
            'compare_sec': self.runtime_diff,
            'compare_gflops': self.gflops_diff,
            # TODO: 'ai': self.arithmetic_intensity,
        }

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        run_cmd = ' '.join(self.job.launcher.command(self.job))
        self.clk_rpt = '%s_CLK.rpt' % self.target_executable
        self.DPscalar_rpt = '%s_DP_scalar.rpt' % self.target_executable
        self.DP128B_rpt = '%s_DP_128B.rpt' % self.target_executable
        self.DP256B_rpt = '%s_DP_256B.rpt' % self.target_executable
        perf_metrics = [
            ('CPU_CLK_UNHALTED.THREAD', self.clk_rpt),
            ('FP_ARITH_INST_RETIRED.SCALAR_DOUBLE', self.DPscalar_rpt),
            ('FP_ARITH_INST_RETIRED.128B_PACKED_DOUBLE', self.DP128B_rpt),
            ('FP_ARITH_INST_RETIRED.256B_PACKED_DOUBLE', self.DP256B_rpt)]
        self.post_run = []
        for perf_metric, perf_rpt in perf_metrics:
            self.post_run += [
                '%s %s -report hw-events -group-by=package -r %s.* -column=%s '
                '&> %s' %
                (run_cmd, self.executable, self.roofdir, perf_metric, perf_rpt)
            ]
        partitiontype = partition.fullname.split(':')[1]
        if partitiontype == 'gpu':
            self.job.options = ['--constraint="gpu&perf"']
        elif partitiontype == 'mc':
            self.job.options = ['--constraint="mc&perf"']

    # --- Elapsed Time:
    @property
    @sn.sanity_function
    def runtime_reported(self):
        sec = sn.extractsingle(r'^Elapsed Time: (?P<sec>\S+)s', self.stdout,
                               'sec', float)
        if self.debug:
            print('sec1={}'.format(sec))

        return sec

    @property
    @sn.sanity_function
    def runtime_metric(self):
        # CPU_CLK_UNHALTED.THREAD:
        mclk = sn.extractsingle(r'^package_0\s+(?P<clk>\d+)',
                                self.clk_rpt, 'clk', float)
        # GHz:
        ghz = sn.extractsingle(r'^\s+Average CPU Frequency: (?P<ghz>\S+) GHz',
                               self.stdout, 'ghz', float)
        # 1 Hz = 1 cycle / 1 second
        sec = (mclk * 10**6) / (ghz * 10**9)
        if self.debug:
            print('sec2={}'.format(sec))
        return sec

    @property
    @sn.sanity_function
    def runtime_diff(self):
        sec = self.runtime_reported - self.runtime_metric
        if self.debug:
            print('sec3={}'.format(sec))
        return sec

    # --- GFLOPS/sec:
    @property
    @sn.sanity_function
    def gflops_reported(self):
        gflops = sn.extractsingle(r'^\s+DP GFLOPS: (?P<gflops>\S+)',
                                  self.stdout, 'gflops', float)
        if self.debug:
            print('gflops1={}'.format(gflops))
        return gflops

    @property
    @sn.sanity_function
    def gflops_metric(self):
        # > srun -Cmc,perf -n1 -t1 likwid-perfctr -g FLOPS_DP -H
        # DP MFLOP/s = 1.0E-06*(x*2 + y + z*4)/runtime where:
        #  x = FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE*
        #  y = FP_ARITH_INST_RETIRED_SCALAR_DOUBLE
        #  z = FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE
        #  TODO: check units with:
        #       ^.*Hardware Event Count.*\((?P<unit>\S+)\)
        # amplxe-cl -report hw-events -r roof.dir.nid00406/ -column=?

        # FP_ARITH_INST_RETIRED.128B_PACKED_DOUBLE:
        DP128B = sn.extractsingle(r'^package_0\s+(?P<M>\d+)',
                                  self.DP128B_rpt, 'M', float)
        # FP_ARITH_INST_RETIRED.SCALAR_DOUBLE:
        DPscalar = sn.extractsingle(r'^package_0\s+(?P<M>\d+)',
                                    self.DPscalar_rpt, 'M', float)
        # FP_ARITH_INST_RETIRED.256B_PACKED_DOUBLE:
        DP256B = sn.extractsingle(r'^package_0\s+(?P<M>\d+)',
                                  self.DP256B_rpt, 'M', float)

        mflops = (DP128B*2 + DPscalar + DP256B*4) / self.runtime_reported
        gflops = mflops / 10**3
        if self.debug:
            print('DP128B={}'.format(DP128B))
            print('DPscalar={}'.format(DPscalar))
            print('DP256B={}'.format(DP256B))
            print('runtime={}'.format(self.runtime_reported))
            print('gflops2={}'.format(gflops))
        return gflops

    @property
    @sn.sanity_function
    def gflops_diff(self):
        gflops = self.gflops_reported - self.gflops_metric
        if self.debug:
            print('gflops3={}'.format(gflops))
        return gflops

    # NOTE: Bandwidth data is missing for a full roofline model.
    # Other tools (advisor, likwid, sde) may help:
    #  > srun -Cmc,perf -n1 -t1 likwid-perfctr -g MEM -H
    #  Memory bandwidth [MBytes/s] = 1.0E-06*(SUM(MBOXxC0) +
    #                                         SUM(MBOXxC1))*64.0/runtime
    #  Memory data volume [GBytes] = 1.0E-09*(SUM(MBOXxC0) +
    #                                         SUM(MBOXxC1))*64.0
    #
    # > srun -Cmc,perf -t1 -n1 likwid-perfctr -g L2 -H
    # L2 bandwidth [MBytes/s] = 1.0E-06*(L1D_REPLACEMENT + L2_TRANS_L1D_WB +
    #                                    ICACHE_MISSES)*64.0/time
    # L2 data volume [GBytes] = 1.0E-09*(L1D_REPLACEMENT + L2_TRANS_L1D_WB +
    #                                    ICACHE_MISSES)*64.0
    #
    # > srun -Cmc,perf -t1 -n1 likwid-perfctr -g L3 -H
    # L3 bandwidth [MBytes/s] = 1.0E-06*(L2_LINES_IN_ALL +
    #                                    L2_LINES_OUT_DEMAND_DIRTY)*64/time
    # L3 data volume [GBytes] = 1.0E-09*(L2_LINES_IN_ALL +
    #                                    L2_LINES_OUT_DEMAND_DIRTY)*64
    #
    # > srun -Cmc,perf -t1 -n1 likwid-perfctr -g CACHES -H
    # Memory bandwidth [MBytes/s] = 1.0E-06*(SUM(CAS_COUNT_RD) +
    #                                        SUM(CAS_COUNT_WR))*64.0/time
    # Memory data volume [GBytes] = 1.0E-09*(SUM(CAS_COUNT_RD) +
    # Vtune supported hw-events:
    # -------
    # Hardware Event Count:CPU_CLK_UNHALTED.THREAD (K)
    # Hardware Event Count:CPU_CLK_UNHALTED.REF_TSC (K)
    # Hardware Event Count:INST_RETIRED.ANY (K)
    # Hardware Event Count:CYCLE_ACTIVITY.STALLS_L1D_MISS (K)
    # Hardware Event Count:CPU_CLK_UNHALTED.REF_XCLK (K)
    # Hardware Event Count:CPU_CLK_UNHALTED.ONE_THREAD_ACTIVE (K)
    # Hardware Event Count:CYCLE_ACTIVITY.STALLS_L2_MISS (K)
    # Hardware Event Count:CYCLE_ACTIVITY.STALLS_MEM_ANY (K)
    # Hardware Event Count:CYCLE_ACTIVITY.STALLS_TOTAL (K)
    # Hardware Event Count:IDQ_UOPS_NOT_DELIVERED.CORE (K)
    # Hardware Event Count:INT_MISC.RECOVERY_CYCLES (K)
    # Hardware Event Count:MEM_LOAD_UOPS_RETIRED.L3_HIT_PS (K)
    # Hardware Event Count:MEM_LOAD_UOPS_RETIRED.L3_MISS_PS (K)
    # Hardware Event Count:RESOURCE_STALLS.SB (K)
    # Hardware Event Count:RS_EVENTS.EMPTY_CYCLES (K)
    # Hardware Event Count:UOPS_EXECUTED.CYCLES_GE_1_UOP_EXEC (K)
    # Hardware Event Count:UOPS_EXECUTED.CYCLES_GE_2_UOPS_EXEC (K)
    # Hardware Event Count:UOPS_EXECUTED.CYCLES_GE_3_UOPS_EXEC (K)
    # Hardware Event Count:UOPS_EXECUTED.CORE:cmask=1 (K)
    # Hardware Event Count:UOPS_EXECUTED.CORE:cmask=2 (K)
    # Hardware Event Count:UOPS_EXECUTED.CORE:cmask=3 (K)
    # Hardware Event Count:UOPS_ISSUED.ANY (K)
    # Hardware Event Count:UOPS_RETIRED.RETIRE_SLOTS (K)
    # Hardware Event Count:IDQ_UOPS_NOT_DELIVERED.CYCLES_0_UOPS_DELIV.CORE (K)
    # Hardware Event Count:OFFCORE_REQUESTS_OUTSTANDING.ALL_DATA_RD:cmask=4 (K)
    # Hardware Event Count:OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DATA_RD (K)
    # Hardware Event Count:MEM_UOPS_RETIRED.ALL_LOADS_PS (K)
    # Hardware Event Count:MEM_UOPS_RETIRED.ALL_STORES_PS (K)
    # Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM_PS (K)
    # Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_DRAM_PS (K)
    # Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_HITM_PS (K)
    # Hardware Event Count:MEM_LOAD_UOPS_L3_MISS_RETIRED.REMOTE_FWD_PS (K)
    # Hardware Event Count:FP_ARITH_INST_RETIRED.SCALAR_SINGLE (K)
    # Hardware Event Count:FP_ARITH_INST_RETIRED.128B_PACKED_SINGLE (K)
    # Hardware Event Count:FP_ARITH_INST_RETIRED.256B_PACKED_SINGLE (K)
    # Hardware Event Count:FP_ARITH_INST_RETIRED.SCALAR_DOUBLE (K)
    # Hardware Event Count:FP_ARITH_INST_RETIRED.128B_PACKED_DOUBLE (K)
    # Hardware Event Count:FP_ARITH_INST_RETIRED.256B_PACKED_DOUBLE (K)
    # Hardware Event Count:INST_RETIRED.X87 (K)
