# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SSHLoginEnvCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = ('Check the values of a set of environment variables '
                      'when accessing remotely over SSH')
        self.valid_systems = ['daint:login', 'dom:login']
        self.sourcesdir = None
        self.valid_prog_environs = ['PrgEnv-cray']
        reference = {
            'CRAY_CPU_TARGET': ('haswell'),
            'CRAYPE_NETWORK_TARGET': 'aries',
            'MODULEPATH': r'[\S+]',
            'MODULESHOME': r'/opt/cray/pe/modules/[\d+\.+]',
            'PE_PRODUCT_LIST': ('CRAYPE_HASWELL:CRAY_RCA:CRAY_ALPS:DVS:'
                                'CRAY_XPMEM:CRAY_DMAPP:CRAY_PMI:CRAY_UGNI:'
                                'CRAY_UDREG:CRAY_LIBSCI:CRAYPE:CRAY:'
                                'PERFTOOLS:CRAYPAT'),
            'SCRATCH': r'/scratch/[\S+]',
            'XDG_RUNTIME_DIR': r'/run/user/[\d+]'
        }
        self.executable = 'ssh'
        echo_args = ' '.join('{0}=${0}'.format(i) for i in reference.keys())
        self.executable_opts = [self.current_system.name,
                                'echo', "'%s'" % echo_args]
        self.sanity_patterns = sn.all(
            sn.map(self.assert_envvar, list(reference.items())))
        self.maintainers = ['RS', 'LM']
        self.tags = {'maintenance', 'production', 'craype'}

    def assert_envvar(self, v):
        return sn.assert_found('='.join(v), self.stdout)
