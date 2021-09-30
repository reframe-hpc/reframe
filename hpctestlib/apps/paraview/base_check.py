# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class ParaView_BaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the ParaView Test.

    ParaView is an open-source, multi-platform data analysis and
    visualization application. ParaView users can quickly build
    visualizations to analyze their data using qualitative and
    quantitative techniques. The data exploration can be done
    interactively in 3D or programmatically using ParaView’s batch
    processing capabilities. ParaView was developed to analyze
    extremely large datasets using distributed memory computing
    resources. It can be run on supercomputers to analyze datasets
    of petascale as well as on laptops for smaller data. ParaView is
    an application framework as well as a turn-key application.
    (see paraview.org)

    The presented abstract run-only class checks the ParaView perfomance.
    To do this, it is necessary to define in the tests the name of vendor
    and the renderer for corresponding platform. This information is used to
    check for errors at the end of program execution. The test itself
    consists in performing a simple task for ParaView (creating a
    colored sphere). The default assumption is that ParaView is already
    installed on the system under test.
    '''

    num_tasks_per_node = required
    executable = 'pvbatch'
    executable_opts = ['coloredSphere.py']
    mc_vendor = variable(str, value='None')
    mc_renderer = variable(str, value='None')
    gpu_vendor = variable(str, value='None')
    gpu_renderer = variable(str, value='None')

    @sanity_function
    def assert_vendor_renderer(self):
        if self.current_partition.name == 'mc':
            return sn.all([
                sn.assert_found(f'Vendor:   {self.mc_vendor}', self.stdout),
                sn.assert_found(f'Renderer: {self.mc_renderer}', self.stdout)
            ])
        elif self.current_partition.name == 'gpu':
            return sn.all([
                sn.assert_found(f'Vendor:   {self.gpu_vendor}', self.stdout),
                sn.assert_found(f'Renderer: {self.gpu_renderer}', self.stdout)
            ])
