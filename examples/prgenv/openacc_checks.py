import os

from reframe.core.pipeline import RegressionTest
from reframe.utility.functions import standard_threshold


class OpenACCFortranCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('testing_fortran_openacc',
                         os.path.dirname(__file__), **kwargs)

        # Uncomment and adjust for your site
        # self.valid_systems = [ 'sys1', 'sys2' ]

        # Uncomment and set the valid prog. environments for your site
        # self.valid_prog_environs = [ 'PrgEnv-cray', 'PrgEnv-pgi' ]

        self.sourcepath = 'vecAdd_openacc.f90'
        self.num_gpus_per_node = 1

        self.sanity_patterns = {
            '-': {
                'final result:\s+(?P<result>\d+\.?\d*)': [
                    ('result', float,
                     lambda value, **kwargs:
                         standard_threshold(value, (1., -1e-5, 1e-5)))
                ],
            }
        }

        # Uncomment and adjust to load OpenACC
        # self.modules = [ 'craype-accel-nvidia60' ]

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = [ 'me' ]
        # self.tags = { 'example' }

    def setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-cray':
            environ.fflags = '-hacc -hnoomp'
        elif environ.name == 'PrgEnv-pgi':
            environ.fflags = '-acc -ta=tesla:cuda8.0'

        super().setup(partition, environ, **job_opts)


def _get_checks(**kwargs):
    return [OpenACCFortranCheck(**kwargs)]
