# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-pgi',
                                    'PrgEnv-intel', 'PrgEnv-cray_classic']
        self.build_system = 'SingleSource'
        self.sourcepath = 'mpit_vars.c'
        src_ref_files = ['mpit_categories.ref', 'mpit_perf_vars.ref',
                         'mpit_control_vars.ref', self.sourcepath]
        self.prebuild_cmds = ['ln -s src/%s .' % f for f in src_ref_files]
        self.num_tasks_per_node = 1
        self.variables = {'MPITEST_VERBOSE': '1', 'MPICH_VERSION_DISPLAY': '1'}
        self.rpt = 'rpt'
        self.executable_opts = [f'&> {self.rpt}']
        self.maintainers = ['JG']
        self.tags = {'production', 'craype', 'maintenance'}

    @rfm.run_before('compile')
    def cray_linker_workaround(self):
        # NOTE: Workaround for using CCE < 9.1 in CLE7.UP01.PS03 and above
        # See Patch Set README.txt for more details.
        if (self.current_system.name == 'dom' and
            self.current_environ.name.startswith('PrgEnv-cray')):
            self.variables['LINKER_X86_64'] = '/usr/bin/ld'

    @rfm.run_before('sanity')
    def set_sanity(self):
        # 1/ MPI Control Variables:
        # --- extract reference data:
        regex = r'^(?P<vars>MPIR\S+)$'
        ref = os.path.join(self.stagedir, 'mpit_control_vars.ref')
        self.ref_control_vars = sorted(sn.extractall(regex, ref, 'vars'))
        # --- extract runtime data:
        regex = r'^\t(?P<vars>MPIR\S+)\t'
        rpt = os.path.join(self.stagedir, self.rpt)
        self.run_control_vars = sorted(sn.extractall(regex, rpt, 'vars'))
        # 2/ MPI Performance Variables:
        # --- extract reference data:
        regex = r'(?P<vars>\w+)'
        ref = os.path.join(self.stagedir, 'mpit_perf_vars.ref')
        self.ref_perf_vars = sorted(sn.extractall(regex, ref, 'vars'))
        # --- extract runtime data:
        regex = r'^\t(?P<vars>(nem_|rma_)\S+)\t'
        rpt = os.path.join(self.stagedir, self.rpt)
        self.run_perf_vars = sorted(sn.extractall(regex, rpt, 'vars'))
        # 3/ MPI Category:
        # --- extract reference data:
        regex = r'^(?P<category>.*)$'
        ref = os.path.join(self.stagedir, 'mpit_categories.ref')
        ref_cat_vars = sorted(sn.extractall(regex, ref, 'category'))
        self.ref_cat_vars = list(filter(None, ref_cat_vars))
        # --- extract runtime data:
        regex = (r'^(?P<category>Category \w+ has \d+ control variables, \d+'
                 r' performance variables, \d+ subcategories)')
        rpt = os.path.join(self.stagedir, self.rpt)
        self.run_cat_vars = sorted(sn.extractall(regex, rpt, 'category'))
        # 4/ Extracted lists can be compared (when sorted):
        self.sanity_patterns = sn.all([
            sn.assert_eq(self.ref_control_vars, self.run_control_vars),
            sn.assert_eq(self.ref_perf_vars, self.run_perf_vars),
            sn.assert_eq(self.ref_cat_vars, self.run_cat_vars),
        ])
