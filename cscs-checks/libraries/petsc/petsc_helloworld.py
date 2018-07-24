import filecmp
import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['dynamic'], ['static'])
class PetscPoisson2DCheck(rfm.RegressionTest):
    def __init__(self, variant):
        super().__init__()
        self.descr = ('Compile/run PETSc 2D Poisson example with cray-petsc '
                      '(%s linking case)') % variant
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel']
        self.sourcepath = 'poisson2d.c'
        self.modules = ['cray-petsc']
        self.num_tasks = 16
        self.num_tasks_per_node = 8
        self.dynamic = True if variant == 'dynamic' else False
        self.executable_opts = ['-da_grid_x 4', '-da_grid_y 4', '-ksp_monitor']

        norms = sn.extractall(r'\s+\d+\s+KSP Residual norm\s+(?P<norm>\S+)',
                              self.stdout, 'norm', float)

        # Check the final residual norm for convergence
        self.sanity_patterns = sn.assert_lt(norms[-1], 1.0e-5)

        self.tags = {'production'}
        self.maintainers = ['WS', 'AJ', 'TM']

    def compile(self):
        if self.dynamic:
            self.current_environ.cxxflags = '-dynamic'
        super().compile()
