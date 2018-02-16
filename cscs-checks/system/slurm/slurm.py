import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest, RunOnlyRegressionTest


# Base class for Slurm simple binary tests
class SlurmSimpleBaseCheck(RunOnlyRegressionTest):
    def __init__(self, name, num_nodes, **kwargs):
        super().__init__('slurm_%s%s' %
                         (name, '_all' if num_nodes == -1 else ''),
                         os.path.dirname(__file__), **kwargs)

        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops'}

        self.num_tasks = num_nodes
        self.tags.add('production')

        if num_nodes == 1:
            self.tags.add('single-node')

        self.num_tasks_per_node = 1
        self.descr = '%s on %s node(s)' % (name, self.num_tasks)
        self.maintainers = ['RS', 'VK']


# Base class for Slurm tests that require compiling some code
class SlurmCompiledBaseCheck(RegressionTest):
    def __init__(self, name, num_nodes, **kwargs):
        super().__init__('slurm_%s%s' %
                         (name, '_all' if num_nodes == -1 else ''),
                         os.path.dirname(__file__), **kwargs)

        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops'}
        self.num_tasks = num_nodes
        self.tags.add('production')

        if num_nodes == 1:
            self.tags.add('single-node')

        self.num_tasks_per_node = 1
        self.descr = '%s on %s node(s)' % (name, self.num_tasks)
        self.maintainers = ['RS', 'VK']


class HostnameCheck(SlurmSimpleBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('hostname', num_nodes, **kwargs)
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
        super().setup(partition, environ, **job_opts)
        num_matches = sn.count(sn.findall(
            self.hostname_string[partition.fullname], self.stdout))
        self.sanity_patterns = sn.assert_eq(self.num_tasks, num_matches)


class EnvironmentVariableCheck(SlurmSimpleBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('check_environment_variable_passing',
                         num_nodes, **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn']
        self.executable = '/bin/echo'
        self.executable_opts = ['$MY_VAR']
        self.variables = {'MY_VAR': 'TEST123456!'}
        num_matches = sn.count(sn.findall(r'TEST123456!', self.stdout))
        self.sanity_patterns = sn.assert_eq(num_nodes, num_matches)


class RequestLargeMemoryNodeCheck(SlurmSimpleBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('request_large_memory_node',
                         num_nodes, **kwargs)
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


class ConstraintRequestGPU(SlurmSimpleBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('submit_job_to_GPU_node_with_constraint',
                         num_nodes, **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.executable = 'nvidia-smi'
        self.sanity_patterns = sn.assert_found(
            r'NVIDIA-SMI.*Driver Version.*', self.stdout)


class ConstraintRequestGPUSetsGRES(SlurmSimpleBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('submit_job_to_GPU_node_sets_GRES',
                         num_nodes, **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.executable = 'scontrol show job ${SLURM_JOB_ID}'
        self.sanity_patterns = sn.assert_found(r'.*Gres=.*gpu:1.*',
                                               self.stdout)


class ConstraintRequestMC(SlurmSimpleBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('submit_job_to_MC_node_with_constraint',
                         num_nodes, **kwargs)
        self.valid_systems = ['daint:mc', 'dom:mc']
        # This is a basic test that should return the number of CPUs on the
        # system which, on a MC node should be 72
        self.executable = 'lscpu -p |grep -v "^#" -c'
        self.sanity_patterns = sn.assert_found(r'72', self.stdout)


class ConstraintRequestCabinetGrouping(SlurmSimpleBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('submit_job_using_c0_0_node_grouping',
                         num_nodes, **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.executable = 'cat /proc/cray_xt/cname'
        self.sanity_patterns = sn.assert_found(r'c0-0.*', self.stdout)

    # we override setup function to pass additional
    # options to Slurm
    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.options = ['--constraint=c0-0']


class EatMemoryCheck(SlurmCompiledBaseCheck):
    def __init__(self, num_nodes, **kwargs):
        super().__init__('memory_overconsumption_kill', num_nodes, **kwargs)
        self.time_limit = (0, 1, 0)
        self.sourcepath = 'eatmemory.c'
        self.tags.add('mem')
        self.executable_opts = ['4000M']
        self.sanity_patterns = sn.assert_found(r'exceeded memory limit',
                                               self.stderr)

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.options += ['--mem=2000']


def _get_checks(**kwargs):
    return [HostnameCheck(1, **kwargs),
            RequestLargeMemoryNodeCheck(1, **kwargs),
            EatMemoryCheck(1, **kwargs),
            EnvironmentVariableCheck(2, **kwargs),
            ConstraintRequestGPU(1, **kwargs),
            ConstraintRequestMC(1, **kwargs),
            ConstraintRequestCabinetGrouping(1, **kwargs),
            ConstraintRequestGPUSetsGRES(1, **kwargs)]
