import reframe as rfm
import reframe.utility.sanity as sn


class BandwidthBase(rfm.RegressionTest):
    def __init__(self):
        super().__init__()

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.build_system = 'Make'
        self.build_system.flags_from_environ = False
        self.sourcesdir = 'https://github.com/RRZE-HPC/likwid.git'
        self.variables = {
            'LD_LIBRARY_PATH': './lib:$LD_LIBRARY_PATH',
            'PATH': './bin:./sbin:$PATH'
        }

        self.executable = 'bin/likwid-bench'

        self.num_tasks = 1
        self.num_tasks_per_core = 2
        self.system_num_cpus = {
            'daint:mc':  72,
            'daint:gpu': 24,
            'dom:mc':  72,
            'dom:gpu': 24,
        }
        self.system_numa_domains = {
            'daint:mc':  ['S0', 'S1'],
            'daint:gpu': ['S0'],
            'dom:mc':  ['S0', 'S1'],
            'dom:gpu': ['S0'],
        }

        # Test each level at half capacity times nthreads per domain
        self.system_cache_sizes = {
            'daint:mc':  {'L1': '288kB', 'L2': '2304kB', 'L3': '23MB',
                          'memory': '100MB'},
            'daint:gpu': {'L1': '192kB', 'L2': '1536kB', 'L3': '15MB',
                          'memory': '100MB'},
            'dom:mc':    {'L1': '288kB', 'L2': '2304kB', 'L3': '23MB',
                          'memory': '100MB'},
            'dom:gpu':   {'L1': '192kB', 'L2': '1536kB', 'L3': '15MB',
                          'memory': '100MB'},
        }

        self.maintainers = ['SK']
        self.tags = {'diagnostic'}

        bw_pattern = sn.extractsingle(r'MByte/s:\s*(?P<bw>\S+)',
                                      self.stdout, 'bw',  float)

        self.sanity_patterns = sn.assert_ge(bw_pattern, 0.0)
        self.perf_patterns = {
            'bandwidth': bw_pattern
        }

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)

        self.postbuild_cmd = ['make install PREFIX=%s INSTALL_CHOWN='
                              '\'-g csstaff -o sebkelle\'' % self.stagedir]


class AllCores(BandwidthBase):
    def __init__(self):
        super().__init__()

        self.valid_systems = ['daint:mc', 'daint:gpu', 'dom:gpu', 'dom:mc']

    def setup(self, partition, environ, **job_opts):

        self.num_cpus_per_task = self.system_num_cpus[partition.fullname]
        numa_domains = self.system_numa_domains[partition.fullname]

        num_cpu_dom = self.num_cpus_per_task / (len(numa_domains) *
                                                self.num_tasks_per_core)
        # result for daint:mc: '-w S0:100MB:18:1:2 -w S1:100MB:18:1:2'
        # format: -w domain:data_size:nthreads:chunk_size:stride
        # chunk_size and stride affect which cpus from <domain> are selected
        workgroups = ' '.join(['-w %s:%s:%d:1:2' %
                              (dom, self.data_size, num_cpu_dom)
                              for dom in numa_domains])

        self.executable_opts = ['-t %s' % self.kernel_name, workgroups]

        super().setup(partition, environ, **job_opts)


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class CPUBandwidthWrite(AllCores):
    def __init__(self):
        super().__init__()

        self.descr = 'CPU -> main memory write benchmark'
        # the kernel to run in likwid
        self.kernel_name = 'store_mem_avx'
        self.data_size = '100MB'

        self.reference = {
            'daint:gpu': {
                'bandwidth': (40000, -0.1, None, 'MB/s')
            },
            'daint:mc': {
                'bandwidth': (80000, -0.1, None, 'MB/s')
            },
            'dom:gpu': {
                'bandwidth': (40000, -0.1, None, 'MB/s')
            },
            'dom:mc': {
                'bandwidth': (80000, -0.1, None, 'MB/s')
            },
        }


@rfm.required_version('>=2.16-dev0')
@rfm.parameterized_test(['L1'], ['L2'], ['L3'], ['memory'])
class CPUBandwidth(AllCores):
    def __init__(self, mem_level):
        super().__init__()

        self.descr = 'CPU <- %s read benchmark' % mem_level

        # the kernel to run in likwid
        self.kernel_name = 'load_avx'
        self.ml = mem_level

        self.refs = {
            'mc':  {'L1': 5100000, 'L2': 2100000, 'L3': 900000,
                    'memory': 130000},
            'gpu': {'L1': 2100000, 'L2': 900000, 'L3': 360000,
                    'memory': 65000},
        }

        self.reference = {
            'daint:gpu': {
                'bandwidth': (self.refs['gpu'][mem_level], -0.1, None, 'MB/s')
            },
            'daint:mc': {
                'bandwidth': (self.refs['mc'][mem_level], -0.1, None, 'MB/s')
            },
            'dom:gpu': {
                'bandwidth': (self.refs['gpu'][mem_level], -0.1, None, 'MB/s')
            },
            'dom:mc': {
                'bandwidth': (self.refs['mc'][mem_level], -0.1, None, 'MB/s')
            },
        }

    def setup(self, partition, environ, **job_opts):
        self.data_size = self.system_cache_sizes[partition.fullname][self.ml]

        super().setup(partition, environ, **job_opts)


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class CPUBandwidthCrossSocket(BandwidthBase):
    def __init__(self):
        super().__init__()

        self.descr = 'CPU S0 <- main memory S1 read'
        ' CPU S1 <- main memory S0 read'

        self.valid_systems = ['daint:mc', 'dom:mc']

        self.kernel_name = 'load_avx'

        self.reference = {
            'daint:mc': {
                'bandwidth': (56000, -0.1, None, 'MB/s')
            },
            'dom:mc': {
                'bandwidth': (56000, -0.1, None, 'MB/s')
            },
        }

    def setup(self, partition, environ, **job_opts):

        self.num_cpus_per_task = self.system_num_cpus[partition.fullname]
        numa_domains = self.system_numa_domains[partition.fullname]

        num_cpu_dom = self.num_cpus_per_task / (len(numa_domains) *
                                                self.num_tasks_per_core)

        # daint:mc: '-w S0:100MB:18:1:2-0:S1 -w S1:100MB:18:1:2-0:S0'
        # format:
        # -w domain:data_size:nthreads:chunk_size:stride-stream_nr:mem_domain
        # chunk_size and stride affect which cpus from <domain> are selected
        workgroups = ' '.join(['-w %s:100MB:%d:1:2-0:%s' %
                              (dom_cpu, num_cpu_dom, dom_mem)
                              for dom_cpu, dom_mem in
                              zip(numa_domains[:2],
                                  reversed(numa_domains[:2]))])

        self.executable_opts = ['-t %s' % self.kernel_name, workgroups]

        super().setup(partition, environ, **job_opts)
