# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class H5PyTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Test that parallel HDF5 can be used by h5py'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['builtin']
        self.modules = ['h5py']
        self.num_tasks = 4
        expected_dataset = ', '.join(str(i) for i in range(self.num_tasks))
        self.sanity_patterns = sn.assert_found(
            rf'.*DATA\s+{{\s+\(0\): {expected_dataset}\s*}}',
            self.stdout
        )
        self.executable = 'python'
        self.executable_opts = ['h5py_mpi_test.py']
        self.postrun_cmds = ['h5dump parallel_test.hdf5']
        self.maintainers = ['TM']
