# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['1.70.0', '19.10', '2.7'], ['1.70.0', '19.10', '3.6'],
                        ['1.70.0', '20.08', '3.8'])
class BoostCrayGnuPythonTest(rfm.RegressionTest):
    def __init__(self, boost_version, cray_gnu_version, python_version):
        self.descr = (f'Test for Boost-{boost_version} for '
                      f'CrayGnu-{cray_gnu_version} with python '
                      f'{python_version} support')
        python_major, python_minor = python_version.split('.')

        if cray_gnu_version == '20.08':
            self.valid_systems = ['dom:mc', 'dom:gpu']
            python_include_suffix = ''
        else:
            self.valid_systems = ['daint:mc', 'daint:gpu']
            python_include_suffix = 'm' if python_major == '3' else ''

        self.valid_prog_environs = ['builtin']
        self.modules = [f'Boost/{boost_version}-CrayGNU-{cray_gnu_version}-'
                        f'python{python_major}']
        self.executable = f'python{python_major} hello.py'
        self.sanity_patterns = sn.assert_found('hello, world', self.stdout)
        self.variables = {
            'PYTHON_INCLUDE':
                f'include/python{python_version}{python_include_suffix}',
            'PYTHON_BOOST_LIB':
                f'boost_python{python_major}{python_minor}'
        }
        self.maintainers = ['JB', 'AJ']
        self.tags = {'scs', 'production'}
