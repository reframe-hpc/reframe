# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class PetscPoisson2DCheck(rfm.RegressionTest):
    linkage = parameter(['dynamic', 'static'])
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-intel']
    modules = ['cray-petsc']
    build_system = 'SingleSource'
    sourcepath = 'poisson2d.c'
    num_tasks = 16
    num_tasks_per_node = 8
    executable_opts = ['-da_grid_x 4', '-da_grid_y 4', '-ksp_monitor']
    tags = {'production', 'craype'}
    maintainers = ['AJ', 'CB']

    @run_after('init')
    def set_description(self):
        self.descr = (f'Compile/run PETSc 2D Poisson example with cray-petsc '
                      f'({self.linkage} linking)')

    @run_after('setup')
    def set_variables(self):
        self.variables = {'CRAYPE_LINK_TYPE': self.linkage}

    @run_before('compile')
    def prg_intel_workaround(self):
        # FIXME: static compilation yields a link error in case of
        # PrgEnv-intel (Cray Bug #255701) workaround use C++ compiler
        if self.linkage == 'static':
            self.build_system.cc = 'CC'

    @sanity_function
    def assert_convergence(self):
        # Check the final residual norm for convergence
        norm = sn.extractsingle(r'\s+\d+\s+KSP Residual norm\s+(?P<norm>\S+)',
                                self.stdout, 'norm', float, -1)
        return sn.assert_lt(norm, 1.0e-5)
