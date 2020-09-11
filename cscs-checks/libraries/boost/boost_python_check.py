# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.os_ext as os_ext
import reframe.utility.sanity as sn


@rfm.parameterized_test(['1.70.0'])
class BoostPythonBindingsTest(rfm.RegressionTest):
    def __init__(self, boostver):
        self.descr = f'Test for Boost {boostver} with Python bindings'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['builtin']
        cdt_version = os_ext.cray_cdt_version()
        self.modules = [f'Boost/{boostver}-CrayGNU-{cdt_version}-python3']
        self.executable = f'python3 hello.py'
        self.sanity_patterns = sn.assert_found('hello, world', self.stdout)
        version_cmd = ('python3 -c \'import sys; '
                       'ver=sys.version_info; '
                       'print(f"{ver.major}{ver.minor}")\'')
        self.variables = {
            'PYTHON_INCLUDE': '$(python3-config --includes)',
            'PYTHON_BOOST_LIB': f'boost_python$({version_cmd})'
        }
        self.maintainers = ['JB', 'AJ']
        self.tags = {'scs', 'production'}
