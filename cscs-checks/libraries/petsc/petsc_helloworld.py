# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['dynamic'], ['static'])
class PetscPoisson2DCheck(rfm.RegressionTest):
    def __init__(self, linkage):
        self.descr = (f'Compile/run PETSc 2D Poisson example with cray-petsc '
                      f'({linkage} linking)')
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel']
        self.sourcepath = 'poisson2d.c'
        self.modules = ['cray-petsc']
        self.num_tasks = 16
        self.num_tasks_per_node = 8
        self.build_system = 'SingleSource'
        # FIXME: static compilation yields a link error in case of
        # PrgEnv-intel (Cray Bug #255701) workaround use C++ compiler
        if linkage == 'static':
            self.build_system.cc = 'CC'

        self.variables = {'CRAYPE_LINK_TYPE': linkage}
        self.executable_opts = ['-da_grid_x 4', '-da_grid_y 4', '-ksp_monitor']

        # Check the final residual norm for convergence
        norm = sn.extractsingle(r'\s+\d+\s+KSP Residual norm\s+(?P<norm>\S+)',
                                self.stdout, 'norm', float, -1)
        self.sanity_patterns = sn.assert_lt(norm, 1.0e-5)
        self.tags = {'production', 'craype'}
        self.maintainers = ['AJ', 'CB']
