import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class BoostCrayGnuPythonTest(RegressionTest):

    def __init__(self, boost_version, cray_gnu_version, python_version,
                 **kwargs):

        def normalize(s):
            return s.replace('.', '_')

        super().__init__('boost_%s_cray_gnu_%s_python_%s_check' % (
            normalize(boost_version), normalize(cray_gnu_version),
            normalize(python_version)), os.path.dirname(__file__), **kwargs)
        self.descr = ('Test for Boost-%s for CrayGnu-%s with python %s '
                      'support') % (boost_version, cray_gnu_version,
                                    python_version)
        self.valid_systems = ['daint:mc', 'daint:gpu', 'dom:mc', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-cray']
        python_major_version = python_version.split('.')[0]
        self.modules = ['Boost/%s-CrayGNU-%s-python%s' % (
            boost_version, cray_gnu_version, python_major_version)]
        self.executable = 'python%s hello.py' % python_major_version
        self.sanity_patterns = sn.assert_found('hello, world', self.stdout)
        python_include_suffix = 'm' if python_major_version == '3' else ''
        python_lib_suffix = '3' if python_major_version == '3' else ''
        self.variables = {
            'PYTHON_INCLUDE': (r'${PYTHON_PATH}/include/python%s%s') % (
                python_version, python_include_suffix),
            'PYTHON_BOOST_LIB': 'boost_python' + python_lib_suffix
        }
        self.maintainers = ['JB', 'AJ']
        self.tags = {'scs', 'production'}


def _get_checks(**kwargs):
    return [BoostCrayGnuPythonTest('1.65.0', '17.08', '2.7', **kwargs),
            BoostCrayGnuPythonTest('1.65.0', '17.08', '3.5', **kwargs)]
