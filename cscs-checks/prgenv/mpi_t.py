# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MpiTCheck(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'Checks MPI_T control/performance variables/categories'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc',
                              'eiger:mc', 'pilatus:mc']
        self.valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.build_system = 'SingleSource'
        self.sourcesdir = 'src/mpi_t'
        self.sourcepath = 'mpit_vars.c'
        self.num_tasks_per_node = 1
        self.variables = {'MPICH_VERSION_DISPLAY': '1', 'MPITEST_VERBOSE': '1'}
        self.rpt = 'rpt'
        self.executable_opts = [f'&> {self.rpt}']
        self.maintainers = ['JG']
        self.tags = {'production', 'craype', 'maintenance'}

    @rfm.run_before('sanity')
    def set_sanity(self):
        # {{{ 0/ MPICH version:
        # MPI VERSION    : CRAY MPICH version 7.7.15 (ANL base 3.2)
        # MPI VERSION    : CRAY MPICH version 8.0.16.17 (ANL base 3.3)
        # MPI VERSION    : CRAY MPICH version 8.1.4.31 (ANL base 3.4a2)
        regex = r'^MPI VERSION\s+: CRAY MPICH version \S+ \(ANL base (\S+)\)'
        rpt_file = os.path.join(self.stagedir, self.rpt)
        mpich_version = sn.extractsingle(regex, rpt_file, 1)
        reference_files = {
            '3.2': {
                'control': 'mpit_control_vars_32.ref',
                'categories': 'mpit_categories_32.ref',
            },
            '3.3': {
                'control': 'mpit_control_vars_33.ref',
                'categories': 'mpit_categories_33.ref',
            },
            '3.4a2': {
                'control': 'mpit_control_vars_34a2.ref',
                'categories': 'mpit_categories_34a2.ref',
            },
        }
        # }}}
        # {{{ 1/ MPI Control Variables: MPIR_...
        # --- extract reference data:
        regex = r'^(?P<vars>MPIR\S+)$'
        ref_file = os.path.join(
            self.stagedir,
            reference_files[sn.evaluate(mpich_version)]['control']
        )
        self.ref_control_vars = sorted(sn.extractall(regex, ref_file, 'vars'))
        # --- extract runtime data:
        regex = r'^\t(?P<vars>MPIR\S+)\t'
        self.run_control_vars = sorted(sn.extractall(regex, rpt_file, 'vars'))
        # --- debug with:"grep -P '\tMPIR+\S*\t' rpt | awk '{print $1}' | sort"
        # }}}
        # {{{ 2/ MPI Category:
        # --- extract reference data:
        regex = r'^(?P<category>.*)$'
        ref = os.path.join(
            self.stagedir,
            reference_files[sn.evaluate(mpich_version)]['categories']
        )
        ref_cat_vars = sorted(sn.extractall(regex, ref, 'category'))
        self.ref_cat_vars = list(filter(None, ref_cat_vars))
        # --- extract runtime data:
        regex = (r'^(?P<category>Category \w+ has \d+ control variables, \d+'
                 r' performance variables, \d+ subcategories)')
        rpt = os.path.join(self.stagedir, self.rpt)
        self.run_cat_vars = sorted(sn.extractall(regex, rpt, 'category'))
        # --- debug with:"grep Category rpt | sort"
        # }}}
        # {{{ 3/ Extracted lists can be compared (when sorted):
        self.sanity_patterns = sn.all([
            sn.assert_eq(self.ref_control_vars, self.run_control_vars,
                         msg='sanity1 "mpit_control_vars.ref" failed'),
            sn.assert_eq(self.ref_cat_vars, self.run_cat_vars,
                         msg='sanity2 "mpit_categories.ref" failed'),
        ])
        # }}}
