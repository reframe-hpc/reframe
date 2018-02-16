import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class HDF5Test(RegressionTest):
    def __init__(self, lang, linkage, **kwargs):
        super().__init__('hdf5_read_write_%s_%s' % (linkage, lang),
                         os.path.dirname(__file__), **kwargs)

        self.flags = ' -%s ' % linkage
        self.lang_names = {
            'c': 'C',
            'f90': 'Fortran 90'
        }

        self.descr = self.lang_names[lang] + ' HDF5 ' + linkage.capitalize()
        self.sourcepath = 'h5ex_d_chunk.' + lang
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.modules = ['cray-hdf5']
        self.keep_files = ['h5dump_out.txt']
        # C and Fortran write transposed matrix
        if lang == 'c':
            self.sanity_patterns = sn.all([
                sn.assert_found(r'Data as written to disk by hyberslabs',
                                self.stdout),
                sn.assert_found(r'Data as read from disk by hyperslab',
                                self.stdout),
                sn.assert_found(r'\(0,0\): 0, 1, 0, 0, 1, 0, 0, 1,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(0,0\): 0, 1, 0, 0, 1, 0, 0, 1,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(1,0\): 1, 1, 0, 1, 1, 0, 1, 1,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(2,0\): 0, 0, 0, 0, 0, 0, 0, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(3,0\): 0, 1, 0, 0, 1, 0, 0, 1,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(4,0\): 1, 1, 0, 1, 1, 0, 1, 1,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(5,0\): 0, 0, 0, 0, 0, 0, 0, 0',
                                'h5dump_out.txt'),
            ])
        else:
            self.sanity_patterns = sn.all([
                sn.assert_found(r'Data as written to disk by hyberslabs',
                                self.stdout),
                sn.assert_found(r'Data as read from disk by hyperslab',
                                self.stdout),
                sn.assert_found(r'\(0,0\): 0, 1, 0, 0, 1, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(1,0\): 1, 1, 0, 1, 1, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(2,0\): 0, 0, 0, 0, 0, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(3,0\): 0, 1, 0, 0, 1, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(4,0\): 1, 1, 0, 1, 1, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(5,0\): 0, 0, 0, 0, 0, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(6,0\): 0, 1, 0, 0, 1, 0,',
                                'h5dump_out.txt'),
                sn.assert_found(r'\(7,0\): 1, 1, 0, 1, 1, 0',
                                'h5dump_out.txt'),
            ])

        self.num_tasks = 1
        self.num_tasks_per_node = 1

        self.maintainers = ['SO']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.post_run = ['h5dump h5ex_d_chunk.h5 > h5dump_out.txt']

    def compile(self):
        self.current_environ.cflags = self.flags
        self.current_environ.cxxflags = self.flags
        self.current_environ.fflags = self.flags
        super().compile()


def _get_checks(**kwargs):
    ret = []
    for lang in ['c', 'f90']:
        for linkage in ['dynamic', 'static']:
            ret.append(HDF5Test(lang, linkage, **kwargs))

    return ret
