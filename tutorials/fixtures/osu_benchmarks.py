import os
import reframe as rfm
import reframe.utility.sanity as sn


class fetch_osu_benchmarks(rfm.RunOnlyRegressionTest):
    version = variable(str, value='5.6.2')
    executable = 'wget'
    executable_opts = [
        f'http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-{version}.tar.gz'  # noqa: E501
    ]

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class build_osu_benchmarks(rfm.CompileOnlyRegressionTest):
    build_system = 'Autotools'
    build_prefix = variable(str)
    osu_benchmarks = fixture(fetch_osu_benchmarks, scope='session')

    @run_before('compile')
    def prepare_build(self):
        tarball = f'osu-micro-benchmarks-{self.osu_benchmarks.version}.tar.gz'
        self.build_prefix = tarball[:-7]  # remove .tar.gz extension

        fullpath = os.path.join(self.osu_benchmarks.stagedir, tarball)
        self.prebuild_cmds = [
            f'cp {fullpath} {self.stagedir}',
            f'tar xzf {tarball}',
            f'cd {self.build_prefix}'
        ]
        self.build_system.max_concurrency = 8

    @sanity_function
    def validate_build(self):
        # If compilation fails, the test would fail in any case, so nothing to
        # further validate here.
        return True


class OSUBenchmarkTestBase(rfm.RunOnlyRegressionTest):
    '''Base class of OSU benchmarks runtime tests'''

    valid_systems = ['daint:gpu']
    valid_prog_environs = ['gnu', 'pgi', 'intel']
    sourcesdir = None
    num_tasks = 2
    num_tasks_per_node = 1
    osu_binaries = fixture(build_osu_benchmarks, scope='environment')

    @sanity_function
    def validate_test(self):
        return sn.assert_found(r'^8', self.stdout)


@rfm.simple_test
class osu_latency_test(OSUBenchmarkTestBase):
    descr = 'OSU latency test'

    @run_before('run')
    def prepare_run(self):
        self.executable = os.path.join(
            self.osu_binaries.stagedir,
            self.osu_binaries.build_prefix,
            'mpi', 'pt2pt', 'osu_latency'
        )
        self.executable_opts = ['-x', '100', '-i', '1000']

    @performance_function('us')
    def latency(self):
        return sn.extractsingle(r'^8\s+(\S+)', self.stdout, 1, float)


@rfm.simple_test
class osu_bandwidth_test(OSUBenchmarkTestBase):
    descr = 'OSU bandwidth test'

    @run_before('run')
    def prepare_run(self):
        self.executable = os.path.join(
            self.osu_binaries.stagedir,
            self.osu_binaries.build_prefix,
            'mpi', 'pt2pt', 'osu_bw'
        )
        self.executable_opts = ['-x', '100', '-i', '1000']

    @performance_function('MB/s')
    def bandwidth(self):
        return sn.extractsingle(r'^4194304\s+(\S+)',
                                self.stdout, 1, float)


@rfm.simple_test
class osu_allreduce_test(OSUBenchmarkTestBase):
    mpi_tasks = parameter(1 << i for i in range(1, 5))
    descr = 'OSU Allreduce test'

    @run_after('init')
    def set_num_tasks(self):
        self.num_tasks = self.mpi_tasks

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.osu_binaries.stagedir,
            self.osu_binaries.build_prefix,
            'mpi', 'collective', 'osu_allreduce'
        )
        self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']

    @performance_function('us')
    def latency(self):
        return sn.extractsingle(r'^8\s+(\S+)', self.stdout, 1, float)
