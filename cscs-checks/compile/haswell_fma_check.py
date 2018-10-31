import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class HaswellFmaCheck(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'check for avx2 instructions'
        self.valid_systems = ['dom:login', 'daint:login', 'kesch:login']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-cray-nompi', 'PrgEnv-gnu-nompi']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi']
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

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if self.current_system.name == 'kesch':
            if environ.name.startswith('PrgEnv-cray'):
                # Ignore CPATH warning
                self.build_system.cflags += ['-h nomessage=1254']
                self.build_system.cxxflags += ['-h nomessage=1254']
            else:
                self.build_system.cflags += ['-march=native']
                self.build_system.cxxflags += ['-march=native']
                self.build_system.fflags += ['-march=native']

        super().setup(partition, environ, **job_opts)
