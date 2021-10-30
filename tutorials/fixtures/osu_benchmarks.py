import os
import reframe as rfm
import reframe.utility.sanity as sn


class OSUDownloadTest(rfm.RunOnlyRegressionTest):
    osu_version = variable(str, '5.6.2')
    executable = f'wget http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-{osu_version}.tar.gz'  # noqa: E501

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class OSUBuildTest(rfm.CompileOnlyRegressionTest):
    build_system = 'Autotools'
    osu_tarball = fixture(OSUDownloadTest, scope='session')

    @run_before('compile')
    def prepare_build(self):
        tarball = f'osu-micro-benchmarks-{self.osu_tarball.osu_version}.tar.gz'
        fullpath = os.path.join(self.osu_tarball.stagedir, tarball)

        self.prebuild_cmds = [
            f'cp {fullpath} {self.stagedir}',
            f'tar xzf {tarball}'
        ]
        self.build_system.max_concurrency = 8

    @sanity_function
    def validate_build(self):
        return sn.assert_not_found('error', self.stderr)


class OSUBenchmarkTestBase(rfm.RunOnlyRegressionTest):
    '''Base class of OSU benchmarks runtime tests'''

    valid_systems = ['daint:gpu']
    valid_prog_environs = ['gnu', 'pgi', 'intel']
    sourcesdir = None
    num_tasks = 2
    num_tasks_per_node = 1
    osu_binaries = fixture(OSUBuildTest, scope='environment')

    @sanity_function
    def validate_test(self):
        return sn.assert_found(r'^8', self.stdout)


@rfm.simple_test
class OSULatencyTest(OSUBenchmarkTestBase):
    descr = 'OSU latency test'

    @require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            osu_binaries.stagedir
            'mpi', 'pt2pt', 'osu_latency'
        )
        self.executable_opts = ['-x', '100', '-i', '1000']

    @performance_function('us')
    def latency(self):
        return sn.extractsingle(r'^8\s+(\S+)', self.stdout, 1, float)
