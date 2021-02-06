# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class BoostPythonBindingsTest(rfm.RegressionTest):
    def __init__(self):
        self.descr = f'Test for Boost with Python bindings'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'eiger:mc']
        self.valid_prog_environs = ['builtin']
        self.modules = [f'Boost']
        self.executable = f'python3 hello.py'
        self.sanity_patterns = sn.assert_found('hello, world', self.stdout)
        version_cmd = ('python3 -c \'import sys; '
                       'ver=sys.version_info; '
                       'print(f"{ver.major}{ver.minor}")\'')
        self.variables = {
            'PYTHON_INCLUDE': '$(python3-config --includes)',
            'PYTHON_BOOST_LIB': f'boost_python$({version_cmd})'
        }
        self.maintainers = ['TM', 'AJ']
        self.tags = {'scs', 'production'}
