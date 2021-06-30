# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HDF5Test(rfm.RegressionTest):
    lang = parameter(['c', 'f90'])
    linkage = parameter(['static', 'dynamic'])
    lang_names = {
        'c': 'C',
        'f90': 'Fortran 90'
    }
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = []
    modules = ['cray-hdf5']
    keep_files = ['h5dump_out.txt']
    num_tasks = 1
    num_tasks_per_node = 1
    build_system = 'SingleSource'
    postrun_cmds = ['h5dump h5ex_d_chunk.h5 > h5dump_out.txt']
    maintainers = ['SO', 'RS']
    tags = {'production', 'craype', 'health'}

    @run_after('init')
    def add_valid_systems(self):
        self.descr = (self.lang_names[self.lang] + ' HDF5 ' +
                      self.linkage.capitalize())
        self.sourcepath = f'h5ex_d_chunk.{self.lang}'
        if self.linkage == 'dynamic':
            self.valid_systems += ['eiger:mc', 'pilatus:mc']
        self.build_system.ldflags = [f'-{self.linkage}']
        if self.current_system.name in ['eiger', 'pilatus']:
            # no cray-hdf5 as of PE 21.02 with PrgEnv-intel on Eiger and
            # Pilatus
            self.valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray',
                                        'PrgEnv-gnu']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi',
                                        'PrgEnv-nvidia']

    @run_before('sanity')
    def set_sanity(self):
        # C and Fortran write transposed matrix
        if self.lang == 'c':
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
