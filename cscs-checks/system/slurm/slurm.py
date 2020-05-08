# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class SlurmSimpleBaseCheck(rfm.RunOnlyRegressionTest):
    '''Base class for Slurm simple binary tests'''

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn',
                              'arolla:cn', 'arolla:pn',
                              'tsa:cn', 'tsa:pn']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops',
                     'production', 'single-node'}
        self.num_tasks_per_node = 1
        if self.current_system.name in ['arolla', 'kesch', 'tsa']:
            self.exclusive_access = True

        self.maintainers = ['RS', 'VH']


class SlurmCompiledBaseCheck(rfm.RegressionTest):
    '''Base class for Slurm tests that require compiling some code'''

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn',
                              'arolla:cn', 'arolla:pn',
                              'tsa:cn', 'tsa:pn']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.tags = {'slurm', 'maintenance', 'ops',
                     'production', 'single-node'}
        self.num_tasks_per_node = 1
        if self.current_system.name in ['arolla', 'kesch', 'tsa']:
            self.exclusive_access = True

        self.maintainers = ['RS', 'VH']


@rfm.simple_test
class HostnameCheck(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.executable = '/bin/hostname'
        self.hostname_patt = {
            'arolla:cn': r'^arolla-cn\d{3}$',
            'arolla:pn': r'^arolla-pp\d{3}$',
            'tsa:cn': r'^tsa-cn\d{3}$',
            'tsa:pn': r'^tsa-pp\d{3}$',
            'kesch:cn': r'^keschcn-\d{4}$',
            'kesch:pn': r'^keschpn-\d{4}$',
            'daint:gpu': r'^nid\d{5}$',
            'daint:mc': r'^nid\d{5}$',
            'dom:gpu': r'^nid\d{5}$',
            'dom:mc': r'^nid\d{5}$',
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
                              'kesch:cn', 'kesch:pn',
                              'arolla:cn', 'arolla:pn',
                              'tsa:cn', 'tsa:pn']
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
        self.executable_opts = ['hostname']
        self.sanity_patterns = sn.assert_found(
            r'error: You have to specify, at least, what sort of node you '
            r'need: -C gpu for GPU enabled nodes, or -C mc for multicore '
            r'nodes.', self.stderr)


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
        self.job.options += ['--mem=120000']


@rfm.simple_test
class DefaultRequestGPU(SlurmSimpleBaseCheck):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'kesch:cn',
                              'tsa:cn']
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
        constraint = f'--constraint={self.current_partition.name}'
        if cabinet:
            constraint += f'&{cabinet}'

        self.job.options += [constraint]


@rfm.simple_test
class MemoryOverconsumptionCheck(SlurmCompiledBaseCheck):
    def __init__(self):
        super().__init__()
        self.time_limit = '1m'
        self.sourcepath = 'eatmemory.c'
        self.tags.add('mem')
        self.executable_opts = ['4000M']
        self.sanity_patterns = sn.assert_found(
            r'(exceeded memory limit)|(Out Of Memory)', self.stderr
        )

    @rfm.run_before('run')
    def set_memory_limit(self):
        self.job.options += ['--mem=2000']
