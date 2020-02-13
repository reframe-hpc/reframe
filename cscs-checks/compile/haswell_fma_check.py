import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class HaswellFmaCheck(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.descr = 'check for avx2 instructions'
        self.valid_systems = ['dom:login', 'daint:login', 'kesch:login']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = [
                'PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-cray-nompi',
                'PrgEnv-gnu-nompi'
            ]
        else:
            self.valid_prog_environs = [
                'PrgEnv-cray', 'PrgEnv-cray_classic', 'PrgEnv-gnu',
                'PrgEnv-intel', 'PrgEnv-pgi'
            ]
            self.modules = ['craype-haswell']

        self.sourcesdir = 'src/haswell_fma'
        self.build_system = 'Make'
        self.build_system.cflags = ['-O3', '-S']
        self.build_system.cxxflags = ['-O3', '-S']
        self.build_system.fflags = ['-O3', '-S']
        self.sanity_patterns = sn.all([
            sn.assert_found(r'vfmadd', 'vectorize_fma_c.s'),
            sn.assert_found(r'vfmadd', 'vectorize_fma_cplusplus.s'),
            sn.assert_found(r'vfmadd', 'vectorize_fma_ftn.s'),
            sn.assert_not_found('warning|WARNING', self.stderr)
        ])

        self.maintainers = ['AJ', 'CB']
        self.tags = {'production', 'craype'}

    @rfm.run_before('setup')
    def set_flags(self):
        if self.current_system.name == 'kesch':
            if environ.name.startswith('PrgEnv-cray'):
                # Ignore CPATH warning
                self.build_system.cflags += ['-h nomessage=1254']
                self.build_system.cxxflags += ['-h nomessage=1254']
            else:
                self.build_system.cflags += ['-march=native']
                self.build_system.cxxflags += ['-march=native']
                self.build_system.fflags += ['-march=native']
        else:
            if environ.name == 'PrgEnv-cray':
                self.build_system.cflags = ['-Ofast', '-S']
                self.build_system.cxxflags = ['-Ofast', '-S']

