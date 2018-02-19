import filecmp
import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


@sn.sanity_function
def sanity_filecmp(output, reference):
    return sn.assert_true(
        filecmp.cmp(output, reference, shallow=False),
        msg="files are not the same: `%s', `%s'" % (output, reference)
    )


class PetscPoisson2DCheck(RegressionTest):
    def __init__(self, variant, **kwargs):
        super().__init__('petsc_2dpoisson_%s' % variant,
                         os.path.dirname(__file__), **kwargs)
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
        self.executable_opts = ['-da_grid_x 4', '-da_grid_y 4', '-mat_view',
                                '> petsc_poisson2d.out']

        self.maintainers = ['WS', 'AJ']
        self.tags = {'production'}

        self.sanity_patterns = sanity_filecmp(
            'petsc_poisson2d.out', 'petsc_poisson2d.ref')

    def compile(self):
        if self.dynamic:
            self.current_environ.cxxflags = '-dynamic'
        super().compile()


def _get_checks(**kwargs):
    return [PetscPoisson2DCheck('dynamic', **kwargs),
            PetscPoisson2DCheck('static',  **kwargs)]
