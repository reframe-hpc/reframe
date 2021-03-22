# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
import re

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext


class FileSystemCommandCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        # TODO: test from cn as well
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['builtin']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'real (?P<real_time>\S+)',
                                          self.stderr, 'real_time', float)
        }
        self.executable = 'time -p'
        self.tags = {'ops', 'diagnostic', 'health'}
        self.maintainers = ['CB', 'VH']

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_eq(self.job.exitcode, 0)


# TODO: if we test only one scratch space we don't need a parameter
@rfm.simple_test
class FileSystemChangeDirCheck(FileSystemCommandCheck):
    directory = parameter(['SCRATCH'])

    def __init__(self):
        super().__init__()
        self.reference = {
            'daint:login': {
                'real_time': (0.1, None, 0.1, 's')
            },
            'dom:login': {
                'real_time': (0.1, None, 0.1, 's')
            }
        }
        self.executable_opts = ['cd', osext.expandvars('$' + self.directory)]


@rfm.simple_test
class FileSystemLsDirCheck(FileSystemCommandCheck):
    directory = parameter(['SCRATCH'])

    def __init__(self):
        super().__init__()
        self.reference = {
            'daint:login': {
                'real_time': (0.1, None, 0.1, 's')
            },
            'dom:login': {
                'real_time': (0.1, None, 0.1, 's')
            }
        }
        self.executable_opts = ['/usr/bin/ls',
                                osext.expandvars('$' + self.directory)]


# TODO: PROJECT is empty
@rfm.simple_test
class FileSystemDuDirCheck(FileSystemCommandCheck):
    directory = parameter(['PROJECT',
                           'HOME',
                           'SCRATCH'])

    def __init__(self):
        super().__init__()
        # TODO: is it possible to append a pattern?
        self.directory_name = osext.expandvars('$' + self.directory)
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'real (?P<real_time>\S+)',
                                          self.stderr, 'real_time', float),
            'size': sn.extractsingle(
                r'(?P<size>\S+).+'+re.escape(self.directory_name),
                self.stdout, 'size', float)
        }

        # TODO: system is not always relevant
        self.reference = {
            'PROJECT': {
                'size': (1000, None, 0.1, 'MB'),
                'real_time': (5.0, None, 0.1, 's')
            },
            'HOME': {
                'size': (900, None, 0.1, 'MB'),
                'real_time': (5.0, None, 0.1, 's')
            },
            'SCRATCH': {
                'size': (900, None, 0.1, 'MB'),
                'real_time': (5.0, None, 0.1, 's')
            }
        }
        self.executable_opts = ['/usr/bin/du -mhs --block-size=1M',
                                self.directory_name]

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(self.directory_name,
                                               self.stdout)


@rfm.simple_test
class FileSystemTouchFileCheck(FileSystemCommandCheck):
    directory = parameter(['SCRATCH'])

    def __init__(self):
        super().__init__()
        self.test_file = osext.expandvars('$' + self.directory +
                                          '/reframe_touch_test_file')
        self.executable_opts = ['touch', self.test_file]

    @rfm.run_after('run')
    def delete_test_file(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)


@rfm.simple_test
class FileSystemCatCheck(FileSystemCommandCheck):
    file = parameter(['/apps/daint/system/etc/BatchDisabled',
                      '/etc/opt/slurm/cgroup.conf',
                      '/etc/opt/slurm/plugstack.conf',
                      '/etc/opt/slurm/slurm.conf',
                      '/etc/opt/slurm/topology.conf',
                      '/etc/opt/slurm/node_prolog.sh',
                      '/etc/opt/slurm/node_epilog.sh',
                      '/etc/opt/slurm/gres.conf'])

    # TODO: find correct test name
    def __init__(self):
        super().__init__()
        self.reference = {
            'daint:login': {
                # TODO: real times have large variances,
                # do we need a specific refererence for each test?
                'real_time': (0.05, None, 0.1, 's')
            },
            'dom:login': {
                'real_time': (0.05, None, 0.1, 's')
            }
        }
        self.executable_opts = ['cat', self.file, ' > /dev/null']


# TODO: this test is almost identical to the cat one
# TODO: /project/csstaff/jenscscs is temporary to speedup the test
@rfm.simple_test
class FileSystemFindCheck(FileSystemCommandCheck):
    directory = parameter([
        '/project/csstaff/jenscscs',
        '/users/jenscscs',
        '/apps/daint/UES/jenscscs/regression/production/reports'])

    # TODO: find correct test name
    def __init__(self):
        super().__init__()
        # TODO: enable this to hide test to non jenscscs users
#        if getpass.getuser() != jenscscs:
#            self.valid_systems = []
        self.reference = {
            'daint:login': {
                # TODO: real times have large variances,
                # do we need a specific refererence for each test?
                'real_time': (0.01, None, 0.1, 's')
            },
            'dom:login': {
                'real_time': (0.01, None, 0.1, 's')
            }
        }
        self.executable_opts = ['find', self.directory,
                                ' -maxdepth 1 > /dev/null']
