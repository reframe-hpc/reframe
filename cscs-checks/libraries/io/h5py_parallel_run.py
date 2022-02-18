# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class H5PyTest(rfm.RunOnlyRegressionTest):
    descr = 'Test that parallel HDF5 can be used by h5py'
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['builtin']
    modules = ['h5py']
    num_tasks = 4
    executable = 'python'
    executable_opts = ['h5py_mpi_test.py']
    postrun_cmds = ['h5dump parallel_test.hdf5']
    tags = {'health', 'production'}
    maintainers = ['TM']

    @sanity_function
    def assert_success(self):
        expected_dataset = ', '.join(str(i) for i in range(self.num_tasks))
        return sn.assert_found(
            rf'.*DATA\s+{{\s+\(0\): {expected_dataset}\s*}}',
            self.stdout
        )
