import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(*([lang, linkage] for lang in ['c', 'f90']
                          for linkage in ['static', 'dynamic']))
class HDF5Test(rfm.RegressionTest):
    def __init__(self, lang, linkage):
        super().__init__()
        lang_names = {
            'c': 'C',
            'f90': 'Fortran 90'
        }
        self.descr = lang_names[lang] + ' HDF5 ' + linkage.capitalize()
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
        self.build_system = 'SingleSource'
        self.build_system.ldflags = ['-%s' % linkage]
        self.post_run = ['h5dump h5ex_d_chunk.h5 > h5dump_out.txt']

        self.maintainers = ['SO']
        self.tags = {'production'}
