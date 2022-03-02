# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class mpit_check(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'Checks MPI_T control/performance variables/categories'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc',
                              'eiger:mc', 'pilatus:mc']
        self.valid_prog_environs = [
            'PrgEnv-aocc', 'PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-intel',
            'PrgEnv-pgi', 'PrgEnv-nvidia']
        self.build_system = 'SingleSource'
        self.sourcesdir = 'src/mpi_t'
        self.sourcepath = 'mpit_vars.c'
        self.prebuild_cmds = ['module list']
        self.variables = {'MPICH_VERSION_DISPLAY': '1', 'MPITEST_VERBOSE': '1'}
        self.num_tasks_per_node = 1
        self.rpt = 'rpt'
        self.executable_opts = [f'&> {self.rpt}']
        self.maintainers = ['JG']
        self.tags = {'craype', 'maintenance'}

    @run_before('sanity')
    def set_sanity(self):
        rpt_file = os.path.join(self.stagedir, self.rpt)
        reference_files = {
            '7.7': {
                'control': 'ref/mpit_control_vars_7.7.ref',
                'categories': 'ref/mpit_categories_7.7.ref',
            },
            '8.1.4': {
                'control': 'ref/mpit_control_vars_8.1.4.ref',
                'categories': 'ref/mpit_categories_8.1.4.ref',
            },
            '8.1.5': {
                'control': 'ref/mpit_control_vars_8.1.5.ref',
                'categories': 'ref/mpit_categories_8.1.5.ref',
            },
            '8.1.6': {
                'control': 'ref/mpit_control_vars_8.1.5.ref',
                'categories': 'ref/mpit_categories_8.1.5.ref',
            },
            '8.1.12': {
                'control': 'ref/mpit_control_vars_8.1.12.ref',
                'categories': 'ref/mpit_categories_8.1.12.ref',
            },
        }
        # {{{ 0/ MPICH version:
        # MPI VERSION  : CRAY MPICH version 7.7.15 (ANL base 3.2)
        # MPI VERSION  : CRAY MPICH version 8.0.16.17 (ANL base 3.3)
        # MPI VERSION  : CRAY MPICH version 8.1.4.31 (ANL base 3.4a2)
        # MPI VERSION  : CRAY MPICH version 8.1.5.32 (ANL base 3.4a2)
        #
        # cdt/21.02 cray-mpich/7.7.16
        # cdt/21.05 cray-mpich/7.7.17
        #
        # cpe/21.04 cray-mpich/8.1.4
        # cpe/21.05 cray-mpich/8.1.5
        # cpe/21.06 cray-mpich/8.1.6
        # cpe/21.08 cray-mpich/8.1.8
        # cpe/21.12 cray-mpich/8.1.12
        regex = r'^MPI VERSION\s+: CRAY MPICH version (\S+) \(ANL base \S+\)'
        mpich_version_major = sn.extractsingle_s(
            r'^(\d+)\.\d+\.\d+', sn.extractsingle(regex, rpt_file, 1), 1
        )
        if mpich_version_major == '7':
            mpich_version_minor = sn.extractsingle_s(
                r'^\d+(\.\d+)\.\d+', sn.extractsingle(regex, rpt_file, 1), 1
            )
        elif mpich_version_major == '8':
            mpich_version_minor = sn.extractsingle_s(
                r'^\d+(\.\d+\.\d+)', sn.extractsingle(regex, rpt_file, 1), 1
            )

        mpich_version = mpich_version_major + mpich_version_minor
        ref_ctrl_file = os.path.join(
            self.stagedir,
            reference_files[sn.evaluate(mpich_version)]['control'])
        ref_catg_file = os.path.join(
            self.stagedir,
            reference_files[sn.evaluate(mpich_version)]['categories'])
        # }}}
        # {{{ 1/ MPI Control Variables: MPIR_...
        # --- extract runtime data:
        regex = r'^\t(?P<vars>MPIR\S+)\t'
        self.run_control_vars = sorted(sn.extractall(regex, rpt_file, 'vars'))
        # --- debug with:"grep -P '\tMPIR+\S*\t' rpt |awk '{print $1}' |sort"
        # --- extract reference data:
        regex = r'^(?P<vars>MPIR\S+)$'
        self.ref_control_vars = sorted(sn.extractall(regex, ref_ctrl_file,
                                                     'vars'))
        # compare runtime and reference in self.sanity_patterns below
        # }}}
        # {{{ 2/ MPI Category:
        # --- extract runtime data:
        regex = (r'^(?P<category>Category \w+ has \d+ control variables, \d+'
                 r' performance variables, \d+ subcategories)')
        self.run_cat_vars = sorted(sn.extractall(regex, rpt_file, 'category'))
        # --- extract reference data:
        regex = r'^(?P<category>.*)$'
        ref_cat_vars = sorted(sn.extractall(regex, ref_catg_file, 'category'))
        self.ref_cat_vars = list(filter(None, ref_cat_vars))
        # compare runtime and reference in self.sanity_patterns below
        # --- debug with:"grep Category rpt |sort"
        # }}}
        # {{{ 3/ Extracted lists can be compared (when sorted):
        self.sanity_patterns = sn.all([
            sn.assert_eq(self.ref_control_vars, self.run_control_vars,
                         msg='sanity1 "mpit_control_vars.ref" failed'),
            sn.assert_eq(self.ref_cat_vars, self.run_cat_vars,
                         msg='sanity2 "mpit_categories.ref" failed'),
        ])
        # }}}
