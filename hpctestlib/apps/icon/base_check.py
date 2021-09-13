# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class RRTMGPTest_Base(rfm.RegressionTest):
    '''Base class for the ICON(Icosahedral Nonhydrostatic Weather
    and Climate Model) Test.

    The ICON modelling framework is a joint project between the German
    Weather Service and the Max Planck Institute for Meteorology for
    developing a unified next-generation global numerical weather prediction
    and climate modelling system. The ICON model has been introduced into
    DWD's operational forecast system in January 2015.
    (see code.mpimet.mpg.de/projects/iconpublic)

    The presented abstract run-only class checks the ICON perfomance.
    The test is verified by checking the performance of the RRTMGP model.
    RRTMGP uses a k-distribution to provide an optical description
    (absorption and possibly Rayleigh optical depth) of the gaseous
    atmosphere, along with the relevant source functions, on a pre-determined
    spectral grid given temperatures, pressures, and gas concentration.
    (see github.com/earth-system-radiation/rte-rrtmgp)
    The default assumption is that ICON is already installed on the
    device under test.
    !!!ATTENTION!!!
    This test is outdated and broken in general and needs to repair.
    '''

    executable = 'python'

    #: This information is used to determine the name of the test file.
    #:
    #: :default: :class:`dict`
    variables = {
        'NCOL': '500',
        'INIFILE': 'openacc-solvers-lw'
    }
    tags = {'external-resources'}

    @run_before('run')
    def set_executable_opts(self):
        self.executable_opts = [
            'util/scripts/run_tests.py',
            '--verbose', '--rel_diff_cut 1e-13',
            '--root ..', '--test ${INIFILE}_ncol-${NCOL}.ini'
        ]

    @run_before('run')
    def set_prerun_cmds(self):
        self.prerun_cmds = [
            'pwd',
            'module load netcdf-python/1.4.1-CrayGNU-19.06-python2',
            'cd test'
            ]

    @sanity_function
    def sanity_pattern(self):
        values = sn.extractall(r'.*\[\S+, (\S+)\]', self.stdout, 1, float)
        return sn.all(
            sn.chain(
                [sn.assert_gt(sn.count(values), 0, msg='regex not matched')],
                sn.map(lambda x: sn.assert_lt(x, 1e-5), values))
        )
