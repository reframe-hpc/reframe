import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['dynamic'], ['static'])
class PetscPoisson2DCheck(rfm.RegressionTest):
    def __init__(self, variant):
        super().__init__()
        self.descr = ('Compile/run PETSc 2D Poisson example with cray-petsc '
                      '(%s linking)') % variant
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel']
        self.sourcepath = 'poisson2d.c'
        self.modules = ['cray-petsc']
        self.num_tasks = 16
        self.num_tasks_per_node = 8
        self.build_system = 'SingleSource'
        if variant == 'dynamic':
            self.build_system.cflags = ['-dynamic']

        self.executable_opts = ['-da_grid_x 4', '-da_grid_y 4', '-ksp_monitor']

        # Check the final residual norm for convergence
        norm = sn.extractsingle(r'\s+\d+\s+KSP Residual norm\s+(?P<norm>\S+)',
                                self.stdout, 'norm', float, -1)
        self.sanity_patterns = sn.assert_lt(norm, 1.0e-5)

        self.tags = {'production'}
        self.maintainers = ['WS', 'AJ', 'TM']
