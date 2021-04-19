# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import getpass
import os
import re

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


class FileSystemGlobal(rfm.RegressionMixin):
    '''Handy class to store common test settings.
    '''
    scratch = parameter(variable(list, value=['SCRATCH']))


class FileSystemCommandCheck(rfm.RunOnlyRegressionTest):
    # TODO: test from cn as well
    valid_systems = ['daint:login', 'dom:login']
    valid_prog_environs = ['builtin']
    executable = 'time -p'
    tags = {'ops', 'diagnostic', 'health'}
    maintainers = ['CB', 'VH']

    @rfm.run_before('sanity')
    def set_sanity_and_perf(self):
        self.sanity_patterns = sn.assert_eq(self.job.exitcode, 0)
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'real (?P<real_time>\S+)',
                                          self.stderr, 'real_time', float)
        }


@rfm.simple_test
class fs_check_cd_dir(FileSystemCommandCheck, FileSystemGlobal):
    reference = {
        'daint:login': {
            'real_time': (0.1, None, 0.1, 's')
        },
        'dom:login': {
            'real_time': (0.1, None, 0.1, 's')
        }
    }

    @rfm.run_before('run')
    def set_executable_ops(self):
        self.executable_opts = ['cd', osext.expandvars(f'${self.scratch}')]


@rfm.simple_test
class fs_check_ls_dir(FileSystemCommandCheck, FileSystemGlobal):
    reference = {
        'daint:login': {
            'real_time': (0.1, None, 0.1, 's')
        },
        'dom:login': {
            'real_time': (0.1, None, 0.1, 's')
        }
    }

    @rfm.run_before('run')
    def set_executable_ops(self):
        self.executable_opts = ['/usr/bin/ls',
                                osext.expandvars(f'${self.scratch}')]


@rfm.simple_test
class fs_check_du_dir(FileSystemCommandCheck, FileSystemGlobal):
    directory = parameter(['PROJECT',
                           'HOME',
                           'SCRATCH'])
    # TODO: system is not always relevant
    reference = {
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

    @rfm.run_before('run')
    def set_executable_ops(self):
        self.path = osext.expandvars(f'${self.directory}')
        self.executable_opts = ['/usr/bin/du -mhs --block-size=1M',
                                self.path]

    @rfm.run_before('sanity')
    def set_sanity_and_perf(self):
        self.sanity_patterns = sn.assert_found(self.path,
                                               self.stdout)
        self.perf_patterns = {
            'real_time': sn.extractsingle(r'real (?P<real_time>\S+)',
                                          self.stderr, 'real_time', float),
            'size': sn.extractsingle(
                r'(?P<size>\S+).+'+re.escape(self.path),
                self.stdout, 'size', float)
        }


@rfm.simple_test
class fs_check_touch_file(FileSystemCommandCheck, FileSystemGlobal):

    @rfm.run_before('run')
    def set_executable_ops(self):
        self.test_file = os.path.join(osext.expandvars(f'${self.scratch}'),
                                      'reframe_touch_test_file')
        self.executable_opts = ['touch', self.test_file]

    @rfm.run_after('run')
    def delete_test_file(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)


@rfm.simple_test
class fs_check_cat_file(FileSystemCommandCheck):
    file = parameter(['/apps/daint/system/etc/BatchDisabled',
                      '/etc/opt/slurm/cgroup.conf',
                      '/etc/opt/slurm/plugstack.conf',
                      '/etc/opt/slurm/slurm.conf',
                      '/etc/opt/slurm/topology.conf',
                      '/etc/opt/slurm/node_prolog.sh',
                      '/etc/opt/slurm/node_epilog.sh',
                      '/etc/opt/slurm/gres.conf'])
    reference = {
        'daint:login': {
            # TODO: real times have large variances,
            # do we need a specific refererence for each test?
            'real_time': (0.05, None, 0.1, 's')
        },
        'dom:login': {
            'real_time': (0.05, None, 0.1, 's')
        }
    }

    @rfm.run_before('run')
    def set_executable_ops(self):
        self.executable_opts = ['cat', self.file, ' > /dev/null']


# TODO: this test is almost identical to the cat one
@rfm.simple_test
class fs_check_find_dir(FileSystemCommandCheck):
    directory = parameter([
        '/project',
        'HOME',
        '/apps/daint/UES/jenscscs/regression/production/reports'])

    @rfm.run_before('run')
    def set_executable_ops(self):
        self.skip_if(getpass.getuser() !=
                     'jenscscs', 'test is valid only for jenscscs user')
        if self.directory is 'HOME':
            self.path = osext.expandvars(f'${self.directory}')
        else:
            self.path = self.directory
        self.executable_opts = ['find', self.path,
                                ' -maxdepth 1 | head -2000 > /dev/null']

    @rfm.run_before('performance')
    def set_perf_reference(self):
        self.reference = {
            '/project': {
                'real_time': (0.01, None, 0.1, 's')
            },
            'HOME': {
                'real_time': (0.01, None, 0.1, 's')
            },
            '/apps/daint/UES/jenscscs/regression/production/reports': {
                'real_time': (0.01, None, 0.1, 's')
            }
        }
