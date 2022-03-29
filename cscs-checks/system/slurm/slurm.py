# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


class SlurmSimpleBaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for Slurm simple binary tests'''

    valid_systems = ['daint:gpu', 'daint:mc',
                     'dom:gpu', 'dom:mc',
                     'arolla:cn', 'arolla:pn',
                     'tsa:cn', 'tsa:pn',
                     'daint:xfer', 'dom:xfer',
                     'eiger:mc', 'pilatus:mc']
    valid_prog_environs = ['PrgEnv-cray']
    tags = {'slurm', 'maintenance', 'ops',
            'production', 'single-node'}
    num_tasks_per_node = 1
    maintainers = ['RS', 'VH']

    @run_after('init')
    def customize_systems(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
            self.exclusive_access = True


class SlurmCompiledBaseCheck(rfm.RegressionTest):
    '''Base class for Slurm tests that require compiling some code'''

    valid_systems = ['daint:gpu', 'daint:mc',
                     'dom:gpu', 'dom:mc']
    valid_prog_environs = ['PrgEnv-cray']
    tags = {'slurm', 'maintenance', 'ops',
            'production', 'single-node'}
    num_tasks_per_node = 1
    maintainers = ['RS', 'VH']


@rfm.simple_test
class HostnameCheck(SlurmSimpleBaseCheck):
    executable = '/bin/hostname'
    valid_prog_environs = ['builtin']
    hostname_patt = {
        'arolla:cn': r'^arolla-cn\d{3}$',
        'arolla:pn': r'^arolla-pp\d{3}$',
        'tsa:cn': r'^tsa-cn\d{3}$',
        'tsa:pn': r'^tsa-pp\d{3}$',
        'daint:gpu': r'^nid\d{5}$',
        'daint:mc': r'^nid\d{5}$',
        'daint:xfer': r'^datamover\d{2}.cscs.ch$',
        'dom:gpu': r'^nid\d{5}$',
        'dom:mc': r'^nid\d{5}$',
        'dom:xfer': r'^nid\d{5}$',
        'eiger:mc': r'^nid\d{6}$',
        'pilatus:mc': r'^nid\d{6}$'
    }

    @run_before('sanity')
    def set_sanity_patterns(self):
        partname = self.current_partition.fullname
        num_matches = sn.count(
            sn.findall(self.hostname_patt[partname], self.stdout)
        )
        self.sanity_patterns = sn.assert_eq(self.num_tasks, num_matches)


@rfm.simple_test
class EnvironmentVariableCheck(SlurmSimpleBaseCheck):
    num_tasks = 2
    valid_systems = ['daint:gpu', 'daint:mc',
                     'dom:gpu', 'dom:mc',
                     'arolla:cn', 'arolla:pn',
                     'tsa:cn', 'tsa:pn',
                     'eiger:mc', 'pilatus:mc']
    executable = '/bin/echo'
    executable_opts = ['$MY_VAR']
    variables = {'MY_VAR': 'TEST123456!'}
    tags.remove('single-node')

    @sanity_function
    def assert_num_tasks(self):
        num_matches = sn.count(sn.findall(r'TEST123456!', self.stdout))
        return sn.assert_eq(self.num_tasks, num_matches)


@rfm.simple_test
class RequiredConstraintCheck(SlurmSimpleBaseCheck):
    valid_systems = ['daint:login', 'dom:login']
    executable = 'srun'
    executable_opts = ['-A', osext.osgroup(), 'hostname']

    @sanity_function
    def assert_found_missing_constraint(self):
        return sn.assert_found(
            r'ERROR: you must specify -C with one of the following: mc,gpu',
            self.stderr
        )


@rfm.simple_test
class RequestLargeMemoryNodeCheck(SlurmSimpleBaseCheck):
    valid_systems = ['daint:mc']
    executable = '/usr/bin/free'
    executable_opts = ['-h']

    @sanity_function
    def assert_memory_is_bounded(self):
        mem_obtained = sn.extractsingle(r'Mem:\s+(?P<mem>\S+)G',
                                        self.stdout, 'mem', float)
        return sn.assert_bounded(mem_obtained, 122.0, 128.0)

    @run_before('run')
    def set_memory_limit(self):
        self.job.options = ['--mem=120000']


@rfm.simple_test
class DefaultRequestGPU(SlurmSimpleBaseCheck):
    valid_systems = ['daint:gpu', 'dom:gpu',
                     'arolla:cn', 'tsa:cn']
    executable = 'nvidia-smi'

    @sanity_function
    def asser_found_nvidia_driver_version(self):
        return sn.assert_found(r'NVIDIA-SMI.*Driver Version.*',
                               self.stdout)


@rfm.simple_test
class DefaultRequestGPUSetsGRES(SlurmSimpleBaseCheck):
    valid_systems = ['daint:gpu', 'dom:gpu']
    executable = 'scontrol show job ${SLURM_JOB_ID}'

    @sanity_function
    def assert_found_resources(self):
        return sn.assert_found(r'.*(TresPerNode|Gres)=.*gpu:1.*', self.stdout)


@rfm.simple_test
class DefaultRequestMC(SlurmSimpleBaseCheck):
    valid_systems = ['daint:mc', 'dom:mc']
    # This is a basic test that should return the number of CPUs on the
    # system which, on a MC node should be 72
    executable = 'lscpu -p |grep -v "^#" -c'

    @sanity_function
    def assert_found_num_cpus(self):
        return sn.assert_found(r'72', self.stdout)


@rfm.simple_test
class ConstraintRequestCabinetGrouping(SlurmSimpleBaseCheck):
    valid_systems = ['daint:gpu', 'daint:mc',
                     'dom:gpu', 'dom:mc']
    executable = 'cat /proc/cray_xt/cname'
    cabinets = {
        'daint:gpu': 'c0-1',
        'daint:mc': 'c1-0',
        # Numbering is inverse in Dom
        'dom:gpu': 'c0-0',
        'dom:mc': 'c0-1',
    }

    @sanity_function
    def assert_found_cabinet(self):
        # We choose a default pattern that will cause assert_found() to fail
        cabinet = self.cabinets.get(self.current_system.name, r'$^')
        return sn.assert_found(fr'{cabinet}.*', self.stdout)

    @run_before('run')
    def set_slurm_constraint(self):
        cabinet = self.cabinets.get(self.current_partition.fullname)
        if cabinet:
            self.job.options = [f'--constraint={cabinet}']


@rfm.simple_test
class MemoryOverconsumptionCheck(SlurmCompiledBaseCheck):
    time_limit = '1m'
    valid_systems += ['eiger:mc', 'pilatus:mc']
    tags.add('mem')
    sourcepath = 'eatmemory.c'
    executable_opts = ['4000M']

    @sanity_function
    def assert_found_exceeded_memory(self):
        return sn.assert_found(r'(exceeded memory limit)|(Out Of Memory)',
                               self.stderr)

    @run_before('run')
    def set_memory_limit(self):
        self.job.options = ['--mem=2000']


@rfm.simple_test
class MemoryOverconsumptionMpiCheck(SlurmCompiledBaseCheck):
    maintainers = ['JG']
    valid_systems += ['eiger:mc', 'pilatus:mc']
    time_limit = '5m'
    sourcepath = 'eatmemory_mpi.c'
    tags.add('mem')
    executable_opts = ['100%']

    @sanity_function
    def assert_found_oom(self):
        return sn.assert_found(r'(oom-kill)|(Killed)',
                               self.stderr)

    @run_before('performance')
    def set_references(self):
        no_limit = (0, None, None, 'GB')
        self.reference = {
            '*': {
                'max_cn_memory': no_limit,
                'max_allocated_memory': (
                    self.reference_meminfo(), -0.05, None, 'GB'
                ),
            }
        }

    @performance_function('GB')
    def max_cn_memory(self):
        return self.reference_meminfo()

    @performance_function('GB')
    def max_allocated_memory(self):
        regex = (r'^Eating \d+ MB\/mpi \*\d+mpi = -\d+ MB memory from \/proc\/'
                 r'meminfo: total: \d+ GB, free: \d+ GB, avail: \d+ GB, using:'
                 r' (\d+) GB')
        return sn.max(sn.extractall(regex, self.stdout, 1, int))

    @run_before('run')
    def set_tasks(self):
        tasks_per_node = {
            'dom:mc': 36,
            'daint:mc': 36,
            'dom:gpu': 12,
            'daint:gpu': 12,
            'eiger:mc': 128,
            'pilatus:mc': 128,
        }
        partname = self.current_partition.fullname
        self.num_tasks_per_node = tasks_per_node[partname]
        self.num_tasks = self.num_tasks_per_node
        self.job.launcher.options = ['-u']

    def reference_meminfo(self):
        reference_meminfo = {
            'dom:gpu': 62,
            'dom:mc': 62,
            'daint:gpu': 62,
            'daint:mc': 62,  # this will pass with 64 GB and above memory sizes
            # this will pass with 256 GB and above memory sizes:
            'eiger:mc': 250,
            'pilatus:mc': 250
        }
        return reference_meminfo[self.current_partition.fullname]


@rfm.simple_test
class slurm_response_check(rfm.RunOnlyRegressionTest):
    command = parameter(['squeue', 'sacct'])
    descr = 'Slurm command test'
    valid_systems = ['daint:login', 'dom:login']
    valid_prog_environs = ['builtin']
    num_tasks = 1
    num_tasks_per_node = 1
    reference = {
        'squeue': {
            'real_time': (0.02, None, 0.1, 's')
        },
        'sacct': {
            'real_time': (0.1, None, 0.1, 's')
        }
    }
    executable = 'time -p'
    tags = {'diagnostic', 'health'}
    maintainers = ['CB', 'VH']

    @run_before('run')
    def set_exec_opts(self):
        self.executable_opts = [self.command]

    @sanity_function
    def assert_exitcode_zero(self):
        return sn.assert_eq(self.job.exitcode, 0)

    @performance_function('s')
    def real_time(self):
        return sn.extractsingle(r'real (?P<real_time>\S+)', self.stderr,
                                'real_time', float)
