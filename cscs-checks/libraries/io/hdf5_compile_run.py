# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class HDF5Test(rfm.RegressionTest):
    lang = parameter(['c', 'f90'])
    linkage = parameter(['static', 'dynamic'])
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    build_system = 'SingleSource'
    modules = ['cray-hdf5']
    keep_files = ['h5dump_out.txt']
    num_tasks = 1
    num_tasks_per_node = 1
    postrun_cmds = ['h5dump h5ex_d_chunk.h5 > h5dump_out.txt']
    maintainers = ['SO', 'RS']
    tags = {'production', 'craype', 'health'}

    @run_after('init')
    def set_description(self):
        lang_names = {
            'c': 'C',
            'f90': 'Fortran 90'
        }
        self.descr = (f'{lang_names[self.lang]} HDF5 '
                      f'{self.linkage.capitalize()}')

    @run_after('init')
    def set_valid_systems(self):
        if self.linkage == 'dynamic':
            self.valid_systems += ['eiger:mc', 'pilatus:mc']

    @run_after('init')
    def set_prog_environs(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            # no cray-hdf5 as of PE 21.02 with PrgEnv-intel on Eiger & Pilatus
            self.valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray',
                                        'PrgEnv-gnu']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi',
                                        'PrgEnv-nvidia']

    @run_after('setup')
    def cdt_2105_skip(self):
        # cray-hdf5 is supported only on PrgEnv-nvidia for cdt >= 21.05
        if self.current_environ.name == 'PrgEnv-nvidia':
            self.skip_if(
                osext.cray_cdt_version() < '21.05',
                "cray-hdf5 is not supported for cdt < 21.05 on PrgEnv-nvidia"
            )
        elif self.current_environ.name == 'PrgEnv-pgi':
            self.skip_if(
                osext.cray_cdt_version() >= '21.05',
                "cray-hdf5 is not supported for cdt >= 21.05 on PrgEnv-pgi"
            )

    @run_before('compile')
    def set_sourcepath(self):
        self.sourcepath = f'h5ex_d_chunk.{self.lang}'

    @run_before('compile')
    def set_ldflags(self):
        self.build_system.ldflags = [f'-{self.linkage}']

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
