# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import getpass
import os
import re

import reframe as rfm
import reframe.utility.sanity as sn


class IorCheck(rfm.RegressionTest):
    base_dir = parameter(['/scratch/e1000',
                          '/scratch/snx3000tds',
                          '/scratch/snx3000',
                          '/scratch/shared/fulen',
                          '/users'])
    username = getpass.getuser()
    time_limit = '5m'

    maintainers = ['SO', 'GLR']
    tags = {'ops', 'production', 'external-resources'}

    @run_after('init')
    def set_description(self):
        self.descr = f'IOR check ({self.base_dir})'

    @run_after('init')
    def extend_tags_based_on_fs(self):
        self.tags |= {self.base_dir}

    @run_after('init')
    def set_fs_information(self):
        self.fs = {
            '/scratch/e1000': {
                'valid_systems': ['eiger:mc', 'pilatus:mc'],
                'eiger': {
                    'num_tasks': 10,
                },
                'pilatus': {
                    'num_tasks': 10,
                }
            },
            '/scratch/snx3000tds': {
                'valid_systems': ['dom:gpu', 'dom:mc'],
                'dom': {
                    'num_tasks': 4,
                }
            },
            '/scratch/snx3000': {
                'valid_systems': ['daint:gpu', 'daint:mc'],
                'daint': {
                    'num_tasks': 10,
                }
            },
            '/users': {
                'valid_systems': ['daint:gpu', 'dom:gpu', 'fulen:normal'],
                'ior_block_size': '8g',
                'daint': {},
                'dom': {},
                'fulen': {
                    'valid_prog_environs': ['PrgEnv-gnu']
                }
            },
            '/scratch/shared/fulen': {
                'valid_systems': ['fulen:normal'],
                'ior_block_size': '48g',
                'fulen': {
                    'num_tasks': 8,
                    'valid_prog_environs': ['PrgEnv-gnu']
                }
            }
        }

        # Setting some default values
        for data in self.fs.values():
            data.setdefault('ior_block_size', '24g')
            data.setdefault('ior_access_type', 'MPIIO')
            data.setdefault(
                'reference',
                {
                    'read_bw': (0, None, None, 'MiB/s'),
                    'write_bw': (0, None, None, 'MiB/s')
                }
            )
            data.setdefault('dummy', {})  # entry for unknown systems

    @run_after('init')
    def set_performance_reference(self):
        # Converting the references from each fs to per system.
        self.reference = {
            '*': self.fs[self.base_dir]['reference']
        }

    @run_after('init')
    def set_valid_systems(self):
        self.valid_systems = self.fs[self.base_dir]['valid_systems']

        cur_sys = self.current_system.name
        if cur_sys not in self.fs[self.base_dir]:
            cur_sys = 'dummy'

        vpe = 'valid_prog_environs'
        penv = self.fs[self.base_dir][cur_sys].get(vpe, ['builtin'])
        self.valid_prog_environs = penv

        tpn = self.fs[self.base_dir][cur_sys].get('num_tasks_per_node', 1)
        self.num_tasks = self.fs[self.base_dir][cur_sys].get('num_tasks', 1)
        self.num_tasks_per_node = tpn

        self.sourcesdir = os.path.join(self.current_system.resourcesdir, 'IOR')

    @run_after('init')
    def set_build_systems(self):
        self.build_system = 'Make'
        self.build_system.options = ['posix', 'mpiio']
        self.build_system.max_concurrency = 1
        self.num_gpus_per_node = 0

    @run_before('run')
    def set_prerun_cmds(self):
        # Default umask is 0022, which generates file permissions -rw-r--r--
        # we want -rw-rw-r-- so we set umask to 0002
        os.umask(2)
        self.test_dir = os.path.join(self.base_dir, self.username, '.ior')
        self.prerun_cmds = ['mkdir -p ' + self.test_dir]

    @run_before('run')
    def set_exec_opts(self):
        self.executable = os.path.join('src', 'C', 'IOR')
        self.test_file = os.path.join(self.test_dir, 'ior')
        self.test_file += '.' + self.current_partition.name
        # executable options depends on the file system
        self.ior_block_size = self.fs[self.base_dir]['ior_block_size']
        self.ior_access_type = self.fs[self.base_dir]['ior_access_type']
        self.executable_opts = ['-B', '-F', '-C ', '-Q 1', '-t 4m', '-D 30',
                                '-b', self.ior_block_size,
                                '-a', self.ior_access_type,
                                '-o', self.test_file]


@rfm.simple_test
class IorWriteCheck(IorCheck):
    executable_opts += ['-w', '-k']
    tags |= {'write'}

    @run_after('init')
    def set_sanity_perf_patterns(self):
        self.sanity_patterns = sn.assert_found(r'^Max Write: ', self.stdout)
        self.perf_patterns = {
            'write_bw': sn.extractsingle(
                r'^Max Write:\s+(?P<write_bw>\S+) MiB/sec', self.stdout,
                'write_bw', float)
        }


class IorReadCheck(IorCheck):
    executable_opts += ['-r']
    tags |= {'read'}

    @run_after('init')
    def set_sanity_perf_patterns(self):
        self.sanity_patterns = sn.assert_found(r'^Max Read: ', self.stdout)
        self.perf_patterns = {
            'read_bw': sn.extractsingle(
                r'^Max Read:\s+(?P<read_bw>\S+) MiB/sec', self.stdout,
                'read_bw', float)
        }

    @run_after('init')
    def set_dependency(self):
        self.depends_on(re.sub(r'IorReadCheck', 'IorWriteCheck', self.name))
