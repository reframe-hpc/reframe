import reframe as rfm
import reframe.utility.sanity as sn


# Base class for Slurm simple binary tests
class SlurmSimpleBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops', 'production'}
        self.num_tasks_per_node = 1
        if self.current_system.name == 'kesch':
            self.exclusive_access = True

        self.maintainers = ['RS', 'VK']

    def setup(self, *args, **kwargs):
        if self.num_tasks == 1:
            self.tags.add('single-node')

        super().setup(*args, **kwargs)

# Base class for Slurm tests that require compiling some code


class SlurmCompiledBaseCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops', 'production'}
        self.num_tasks_per_node = 1
        if self.current_system.name == 'kesch':
            self.exclusive_access = True

        self.maintainers = ['RS', 'VK']

    def setup(self, *args, **kwargs):
        if self.num_tasks == 1:
            self.tags.add('single-node')

        super().setup(*args, **kwargs)


@rfm.simple_test
class HostnameCheck(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.executable = '/bin/hostname'
        self.hostname_string = {
            'kesch:cn': r'keschcn-\d{4}\b',
            'kesch:pn': r'keschpn-\d{4}\b',
            'daint:gpu': r'nid\d{5}\b',
            'daint:mc': r'nid\d{5}\b',
            'dom:gpu': r'nid\d{5}\b',
            'dom:mc': r'nid\d{5}\b',
        }

    def setup(self, partition, environ, **job_opts):
        num_matches = sn.count(sn.findall(
            self.hostname_string[partition.fullname], self.stdout))
        self.sanity_patterns = sn.assert_eq(self.num_tasks, num_matches)
        super().setup(partition, environ, **job_opts)


@rfm.simple_test
class EnvironmentVariableCheck(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.num_tasks = 2
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn']
        self.executable = '/bin/echo'
        self.executable_opts = ['$MY_VAR']
        self.variables = {'MY_VAR': 'TEST123456!'}
        num_matches = sn.count(sn.findall(r'TEST123456!', self.stdout))
        self.sanity_patterns = sn.assert_eq(self.num_tasks, num_matches)


@rfm.simple_test
class RequestLargeMemoryNodeCheck(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:mc']
        self.executable = '/usr/bin/free'
        self.executable_opts = ['-h']
        mem_obtained = sn.extractsingle(r'Mem:\s+(?P<mem>\S+)G',
                                        self.stdout, 'mem', float)
        self.sanity_patterns = sn.assert_bounded(mem_obtained, 122.0, 128.0)

    # we override setup function to pass additional
    # options to Slurm
    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.options += ['--mem=120000']


@rfm.simple_test
class DefaultRequestGPU(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.executable = 'nvidia-smi'
        self.sanity_patterns = sn.assert_found(
            r'NVIDIA-SMI.*Driver Version.*', self.stdout)


@rfm.simple_test
class DefaultRequestGPUSetsGRES(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.executable = 'scontrol show job ${SLURM_JOB_ID}'
        self.sanity_patterns = sn.assert_found(r'.*Gres=.*gpu:1.*',
                                               self.stdout)


@rfm.simple_test
class DefaultRequestMC(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:mc', 'dom:mc']
        # This is a basic test that should return the number of CPUs on the
        # system which, on a MC node should be 72
        self.executable = 'lscpu -p |grep -v "^#" -c'
        self.sanity_patterns = sn.assert_found(r'72', self.stdout)


@rfm.simple_test
class ConstraintRequestCabinetGrouping(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.executable = 'cat /proc/cray_xt/cname'
        self.sanity_patterns = sn.assert_found(r'c0-0.*', self.stdout)

    # we override setup function to pass additional
    # options to Slurm
    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.options = ['--constraint=c0-0']


@rfm.simple_test
class MemoryOverconsumptionCheck(SlurmCompiledBaseCheck):
    def __init__(self):
        super().__init__()
        self.time_limit = (0, 1, 0)
        self.sourcepath = 'eatmemory.c'
        self.tags.add('mem')
        self.executable_opts = ['4000M']
        self.sanity_patterns = sn.assert_found(r'exceeded memory limit',
                                               self.stderr)

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.options += ['--mem=2000']
