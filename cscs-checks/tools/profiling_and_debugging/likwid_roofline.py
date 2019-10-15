import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*[[repeat, toolsversion, datalayout]
                          for repeat in ['600000']
                          for toolsversion in ['4.3.3']
                          # for datalayout in ['G3_AOS_SCALAR']
                          for datalayout in ['G3_AOS_SCALAR', 'G3_SOA_SCALAR',
                                             'G3_AOS_VECTOR', 'G3_SOA_VECTOR']
                          ])
class IntelRooflineLikwidTest(rfm.RegressionTest):
    '''This test checks the values reported by RRZE likwid roofline model:

G3_AOS_SCALAR DP Mflops/sec = 3280.32 L2 bandwidth [MBytes/s] = 39441.3 0.0831697
G3_AOS_VECTOR DP Mflops/sec = 6432.24 L2 bandwidth [MBytes/s] = 76914 0.083629
G3_SOA_SCALAR DP Mflops/sec = 3288.39 L2 bandwidth [MBytes/s] = 9.98179 329.439
G3_SOA_VECTOR DP Mflops/sec = 21126.6 L2 bandwidth [MBytes/s] = 9.6529 2188.63 2.3F/B
                              10GF                              60000  0.18

        > https://crd.lbl.gov/assets/Uploads/ECP18-Roofline-3-LIKWID.pdf
        > likwid-perfctr -g CACHES -H

        > Get group definition with (identical result):
        > cat $EBROOTLIKWID/share/likwid/perfgroups/broadwell/FLOPS_DP.txt
        > srun -Cmc,perf -n1 -t1 likwid-perfctr -g FLOPS_DP -H
        DP MFLOP/s = 1.0E-06*(FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE*2 +
                              FP_ARITH_INST_RETIRED_SCALAR_DOUBLE +
                              FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE*4)
                              /runtime

        > srun -Cmc,perf -n1 -t1 likwid-perfctr -g MEM -H
        Memory bandwidth [MBytes/s] = 1.0E-06*(SUM(MBOXxC0) +
                                               SUM(MBOXxC1))*64.0/runtime
        Memory data volume [GBytes] = 1.0E-09*(SUM(MBOXxC0) +
                                               SUM(MBOXxC1))*64.0

       > srun -Cmc,perf -t1 -n1 likwid-perfctr -g L2 -H
       L2 bandwidth [MBytes/s] = 1.0E-06*(L1D_REPLACEMENT + L2_TRANS_L1D_WB +
                                          ICACHE_MISSES)*64.0/time
       L2 data volume [GBytes] = 1.0E-09*(L1D_REPLACEMENT + L2_TRANS_L1D_WB +
                                          ICACHE_MISSES)*64.0

       > srun -Cmc,perf -t1 -n1 likwid-perfctr -g L3 -H
       L3 bandwidth [MBytes/s] = 1.0E-06*(L2_LINES_IN_ALL +
                                          L2_LINES_OUT_DEMAND_DIRTY)*64/time
       L3 data volume [GBytes] = 1.0E-09*(L2_LINES_IN_ALL +
                                          L2_LINES_OUT_DEMAND_DIRTY)*64

       > srun -Cmc,perf -t1 -n1 likwid-perfctr -g CACHES -H
       Memory bandwidth [MBytes/s] = 1.0E-06*(SUM(CAS_COUNT_RD) +
                                              SUM(CAS_COUNT_WR))*64.0/time
       Memory data volume [GBytes] = 1.0E-09*(SUM(CAS_COUNT_RD) +
                                              SUM(CAS_COUNT_WR))*64.0
    '''
    def __init__(self, repeat, toolsversion, datalayout):
        super().__init__()
        self.descr = 'Roofline Analysis test with Likwid:'
        self.valid_systems = ['dom:mc']
        # Reporting MFLOPS is not available on Intel Haswell cpus, see
        # https://www.intel.fr/content/dam/www/public/us/en/documents/manuals/
        # 64-ia-32-architectures-software-developer-vol-1-manual.pdf
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['likwid']
        # likwid/4.3.3-perf_event
        self.sourcesdir = os.path.join(self.current_system.external-external-resourcesdir,
                                       'roofline', 'intel_advisor')
        self.build_system = 'SingleSource'
        self.sourcepath = '_roofline.cpp'
        self.executable = 'likwid-perfctr'
        self.target_executable = './roof.exe'
        self.build_system.cppflags = ['-D_LIKWID', '-DLIKWID_PERFMON',
                                      '-I$EBROOTLIKWID/include']
        self.prgenv_flags = {
            'PrgEnv-intel': ['-g', '-O2', '-std=c++11', '-restrict'],
            # '-qopt-streaming-stores', 'always',
        }
        self.build_system.ldflags = ['-L$EBROOTLIKWID/lib', '-llikwid']
        self.prebuild_cmd = [
            'patch -s < LIKWID/roofline_template.patch',
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
        ]
        self.tool_flags = ['-C 0 -g FLOPS_DP -m %s ' % self.target_executable]
        # -C 0 : sets processor id(s) to pin threads and measure
        # -g   : sets performance group
        # -m   : use likwid API
        self.executable_opts = self.tool_flags
        self.maintainers = ['JG']
        self.tags = {'scs', 'external-external-resources'}
        # self.rpt = '%s.rpt' % self.target_executable
        self.sanity_patterns = sn.all([
            sn.assert_found('loop complete.', self.stdout),
            sn.assert_eq(sn.extractsingle(
                r'^likwid-perfctr -- Version (?P<toolsversion>\d.\d.\d)',
                self.stdout, 'toolsversion'), toolsversion),
        ])
        # References for Intel Broadwell CPU (E5-2695 v4):
        references = {
            'G3_AOS_SCALAR': {
                'dom:mc': {
                    'gflops': (0.596, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.16, -0.05, 0.05, 'flop/byte')
                }
            },
            'G3_SOA_SCALAR': {
                'dom:mc': {
                    'gflops': (0.612, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.16, -0.05, 0.05, 'flop/byte')
                }
            },
            'G3_AOS_VECTOR': {
                'dom:mc': {
                    'gflops': (1.152, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.125, -0.05, 0.05, 'flop/byte')
                }
            },
            'G3_SOA_VECTOR': {
                'dom:mc': {
                    'gflops': (1.125, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.16, -0.05, 0.05, 'flop/byte')
                }
            },
        }
        self.reference = references[datalayout]
        self.perf_patterns = {
            'gflops': self.gflops,
            'ai': self.arithmetic_intensity,
        }

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        run_cmd = ' '.join(self.job.launcher.command(self.job))
        self.post_run = ['%s -v' % self.executable]
        # self.perf_group = ['L2', 'L3']
        self.perf_group = ['L2', 'L3', 'CACHES', 'DATA',
                           'MEM', 'MEM_DP', 'MEM_SP']
        for perf_group in self.perf_group:
            self.post_run += ['%s %s -C 0 -g %s -m %s' %
                              (run_cmd, self.executable, perf_group,
                               self.target_executable)]
        partitiontype = partition.fullname.split(':')[1]
        if partitiontype == 'gpu':
            self.job.options = ['--constraint="gpu&perf"']
        elif partitiontype == 'mc':
            self.job.options = ['--constraint="mc&perf"']

    @property
    @sn.sanity_function
    def arithmetic_intensity(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        bytes = sn.extractsingle(r'^--->Total Bytes = (?P<bytes>\d+)',
                                 self.rpt, 'bytes', int)
        # debug: print('ai={}'.format(flops/bytes))
        return flops/bytes

    @property
    @sn.sanity_function
    def gflops(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        msec = sn.extractsingle(r'^elapsed time: (?P<msec>\d+)ms', self.stdout,
                                'msec', float)
        # debug: print('gflops={}'.format(flops/((msec/1000)*10**6)))
        return (flops/((msec/1000))/10**9)
