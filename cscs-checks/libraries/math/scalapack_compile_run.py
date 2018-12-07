import os

import reframe as rfm
import reframe.utility.sanity as sn


class ScaLAPACKTest(rfm.RegressionTest):
    def __init__(self, linkage):
        super().__init__()
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'scalapack')
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:mc',
                              'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel']
        self.num_tasks = 16
        self.num_tasks_per_node = 8
        self.variables = {'CRAYPE_LINK_TYPE': linkage}

        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-cray']
            if linkage == 'static':
                # Static linkage not supported on Kesch
                self.valid_prog_environs = []

        self.build_system = 'SingleSource'
        self.build_system.fflags = ['-O3']
        self.maintainers = ['CB', 'LM', 'MKr']
        self.tags = {'production'}


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['static'], ['dynamic'])
class ScaLAPACKSanity(ScaLAPACKTest):
    def __init__(self, linkage):
        super().__init__(linkage)
        self.sourcepath = 'scalapack_compile_run.f'

        def fortran_float(value):
            return float(value.replace('D', 'E'))

        def scalapack_sanity(number1, number2, expected_value):
            symbol = 'z{0}{1}'.format(number1, number2)
            pattern = r'Z\(     {0},     {1}\)=\s+(?P<{2}>\S+)'.format(
                number2, number1, symbol)
            found_value = sn.extractsingle(pattern, self.stdout, symbol,
                                           fortran_float)
            return sn.assert_lt(sn.abs(expected_value - found_value), 1.0e-15)

        self.sanity_patterns = sn.all([
            scalapack_sanity(1, 1, -0.04853779318803846),
            scalapack_sanity(1, 2, -0.12222271866735863),
            scalapack_sanity(1, 3, -0.28248513530339736),
            scalapack_sanity(1, 4, 0.95021462733774853),
            scalapack_sanity(2, 1, 0.09120722270314352),
            scalapack_sanity(2, 2, 0.42662009209279039),
            scalapack_sanity(2, 3, -0.8770383032575241),
            scalapack_sanity(2, 4, -0.2011973015939371),
            scalapack_sanity(3, 1, 0.4951930430455262),
            scalapack_sanity(3, 2, -0.7986420412618930),
            scalapack_sanity(3, 3, -0.2988441319801194),
            scalapack_sanity(3, 4, -0.1662736444220721),
            scalapack_sanity(4, 1, 0.8626176298213052),
            scalapack_sanity(4, 2, 0.4064822185450869),
            scalapack_sanity(4, 3, 0.2483911184660867),
            scalapack_sanity(4, 4, 0.1701907253504270)
        ])


# FIXME: This test is obsolete; it is kept only for reference.
# NOTE:  The test case is very small, but larger cases did not succeed!
@rfm.required_version('>=2.14')
@rfm.parameterized_test(['dynamic'])
class ScaLAPACKPerf(ScaLAPACKTest):
    def __init__(self, linkage):
        super().__init__(linkage)

        self.tags |= {'monch_acceptance'}
        self.sourcepath = 'scalapack_performance_compile_run.f'
        self.valid_systems = ['monch:compute']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 64
        self.num_tasks_per_node = 16

        self.sanity_patterns = sn.assert_found(r'Run', self.stdout)
        self.perf_patterns = {
            'perf': sn.max(
                sn.extractall(r'GFLOPS/s:\s+(?P<gflops>\S+)',
                              self.stdout, 'gflops', float)
            )
        }

        self.reference = {
            'monch:compute': {
                'perf': (24., -0.1, None)
            }
        }
