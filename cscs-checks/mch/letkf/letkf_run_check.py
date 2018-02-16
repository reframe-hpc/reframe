import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RunOnlyRegressionTest


class LETKFRunCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('letkf_run_check', os.path.dirname(__file__),
                         **kwargs)
        self.descr = 'LETKF benchmark; MCH; ' \
                     'simplification of the production code'
        self.valid_systems = ['kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.letkf_prefix = os.path.join(self.current_system.resourcesdir,
                                         'LETKF', 'test_letkf')
        self.executable = os.path.join(self.letkf_prefix, 'bin', 'var3d')
        self.sourcesdir = os.path.join(self.letkf_prefix, 'input')
        self.num_tasks = 264
        self.num_tasks_per_node = 24
        self.use_multithreading = True
        self.sanity_patterns = sn.assert_eq(sn.count(sn.findall(
            r'No differences found.', 'letkf_run_check.out')), 3)

        self.modules = ['ScaLAPACK', 'netCDF-Fortran']
        self.variables = {'NETCDF_PATH': '$EBROOTNETCDF',
                          'NETCDFF_PATH': '$EBROOTNETCDFMINFORTRAN',
                          'MALLOC_MMAP_MAX_': '0',
                          'MALLOC_TRIM_THRESHOLD_': '536870912'}

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}
        self.strict_check = False

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        # additional program call in order to generate the tracing output for
        # the sanity check
        postproc_cmd = '{prog} . {output} .'
        self.job.post_run = [
            postproc_cmd.format(
                prog=os.path.join(self.letkf_prefix, 'bin', 'cmp_feedback'),
                output=os.path.join(self.letkf_prefix, 'reference_output'))
        ]


def _get_checks(**kwargs):
    return [LETKFRunCheck(**kwargs)]
