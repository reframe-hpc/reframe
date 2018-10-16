import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['1.67.0', '18.08', '2.7'], ['1.67.0', '18.08', '3.6'])
class BoostCrayGnuPythonTest(rfm.RegressionTest):
    def __init__(self, boost_version, cray_gnu_version, python_version):
        super().__init__()
        self.descr = ('Test for Boost-%s for CrayGnu-%s with python %s '
                      'support') % (boost_version, cray_gnu_version,
                                    python_version)
        self.valid_systems = ['daint:mc', 'daint:gpu', 'dom:mc', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        python_major, python_minor = python_version.split('.')
        self.modules = ['Boost/%s-CrayGNU-%s-python%s' % (
            boost_version, cray_gnu_version, python_major)]
        self.executable = 'python%s hello.py' % python_major
        self.sanity_patterns = sn.assert_found('hello, world', self.stdout)
        python_include_suffix = 'm' if python_major == '3' else ''
        python_lib = 'boost_python%s%s' % (python_major, python_minor)
        self.variables = {
            'PYTHON_INCLUDE': r'include/python%s%s' % (
                python_version, python_include_suffix),
            'PYTHON_BOOST_LIB': python_lib
        }
        self.maintainers = ['JB', 'AJ']
        self.tags = {'scs', 'production'}
