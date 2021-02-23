# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


class SlurmSimpleBaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for Slurm simple binary tests'''

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'arolla:cn', 'arolla:pn',
                              'tsa:cn', 'tsa:pn',
                              'daint:xfer', 'eiger:mc']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops',
                     'production', 'single-node'}
        self.num_tasks_per_node = 1
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
            self.exclusive_access = True

        self.maintainers = ['RS', 'VH']


class SlurmCompiledBaseCheck(rfm.RegressionTest):
    '''Base class for Slurm tests that require compiling some code'''

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops',
                     'production', 'single-node'}
        self.num_tasks_per_node = 1

        self.maintainers = ['RS', 'VH']


@rfm.simple_test
class HostnameCheck(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.executable = '/bin/hostname'
        self.valid_prog_environs = ['builtin']
        self.hostname_patt = {
            'arolla:cn': r'^arolla-cn\d{3}$',
            'arolla:pn': r'^arolla-pp\d{3}$',
            'tsa:cn': r'^tsa-cn\d{3}$',
            'tsa:pn': r'^tsa-pp\d{3}$',
            'daint:gpu': r'^nid\d{5}$',
            'daint:mc': r'^nid\d{5}$',
            'daint:xfer': r'^datamover\d{2}.cscs.ch$',
            'dom:gpu': r'^nid\d{5}$',
            'dom:mc': r'^nid\d{5}$',
            'eiger:mc': r'^nid\d{6}$',
        }

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        partname = self.current_partition.fullname
        num_matches = sn.count(
            sn.findall(self.hostname_patt[partname], self.stdout)
        )
        self.sanity_patterns = sn.assert_eq(self.num_tasks, num_matches)


@rfm.simple_test
class EnvironmentVariableCheck(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.num_tasks = 2
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'arolla:cn', 'arolla:pn',
                              'tsa:cn', 'tsa:pn',
                              'eiger:mc']
        self.executable = '/bin/echo'
        self.executable_opts = ['$MY_VAR']
        self.variables = {'MY_VAR': 'TEST123456!'}
        self.tags.remove('single-node')
        num_matches = sn.count(sn.findall(r'TEST123456!', self.stdout))
        self.sanity_patterns = sn.assert_eq(self.num_tasks, num_matches)


@rfm.simple_test
class RequiredConstraintCheck(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:login', 'dom:login']
        self.executable = 'srun'
        self.executable_opts = ['-A', osext.osgroup(), 'hostname']
        self.sanity_patterns = sn.assert_found(
            r'ERROR: you must specify -C with one of the following: mc,gpu',
            self.stderr
        )


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

    @rfm.run_before('run')
    def set_memory_limit(self):
        self.job.options = ['--mem=120000']


@rfm.simple_test
class DefaultRequestGPU(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'tsa:cn']
        self.executable = 'nvidia-smi'
        self.sanity_patterns = sn.assert_found(
            r'NVIDIA-SMI.*Driver Version.*', self.stdout)


@rfm.simple_test
class DefaultRequestGPUSetsGRES(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.executable = 'scontrol show job ${SLURM_JOB_ID}'
        self.sanity_patterns = sn.assert_found(
            r'.*(TresPerNode|Gres)=.*gpu:1.*', self.stdout)


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
        self.cabinets = {
            'daint:gpu': 'c0-1',
            'daint:mc': 'c1-0',

            # Numbering is inverse in Dom
            'dom:gpu': 'c0-0',
            'dom:mc': 'c0-1',
        }

        # We choose a default pattern that will cause assert_found() to fail
        cabinet = self.cabinets.get(self.current_system.name, r'$^')
        self.sanity_patterns = sn.assert_found(fr'{cabinet}.*', self.stdout)

    @rfm.run_before('run')
    def set_slurm_constraint(self):
        cabinet = self.cabinets.get(self.current_partition.fullname)
        if cabinet:
            self.job.options = [f'--constraint={cabinet}']


@rfm.simple_test
class MemoryOverconsumptionCheck(SlurmCompiledBaseCheck):
    def __init__(self):
        super().__init__()
        self.time_limit = '1m'
        self.valid_systems += ['eiger:mc']
        self.sourcepath = 'eatmemory.c'
        self.tags.add('mem')
        self.executable_opts = ['4000M']
        self.sanity_patterns = sn.assert_found(
            r'(exceeded memory limit)|(Out Of Memory)', self.stderr
        )

    @rfm.run_before('run')
    def set_memory_limit(self):
        self.job.options = ['--mem=2000']


@rfm.simple_test
class MemoryOverconsumptionMpiCheck(SlurmCompiledBaseCheck):
    def __init__(self):
        super().__init__()
        self.maintainers = ['JG']
        self.valid_systems += ['eiger:mc', 'pilatus:mc']
        self.time_limit = '5m'
        self.sourcepath = 'eatmemory_mpi.c'
        self.tags.add('mem')
        self.executable_opts = ['100%']
        self.sanity_patterns = sn.assert_found(r'(oom-kill)|(Killed)',
                                               self.stderr)
        # {{{ perf
        regex = (r'^Eating \d+ MB\/mpi \*\d+mpi = -\d+ MB memory from \/proc\/'
                 r'meminfo: total: \d+ GB, free: \d+ GB, avail: \d+ GB, using:'
                 r' (\d+) GB')
        self.perf_patterns = {
            'max_cn_memory': sn.getattr(self, 'reference_meminfo'),
            'max_allocated_memory': sn.max(
                sn.extractall(regex, self.stdout, 1, int)
            ),
        }
        no_limit = (0, None, None, 'GB')
        self.reference = {
            '*': {
                'max_cn_memory': no_limit,
                'max_allocated_memory': (
                    sn.getattr(self, 'reference_meminfo'), -0.05, None, 'GB'
                ),
            }
        }
        # }}}

    # {{{ hooks
    @rfm.run_before('run')
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
    # }}}

    @property
    @sn.sanity_function
    def reference_meminfo(self):
        reference_meminfo = {
            'dom:gpu': 62,
            'dom:mc': 62,
            'daint:gpu': 62,
            'daint:mc': 62,  # this will pass with 64 GB and above memory sizes
            # this will pass with 256 GB and above memory sizes:
            'eiger:mc': 250,
            'pilatus:mc': 250,
        }
        return reference_meminfo[self.current_partition.fullname]
