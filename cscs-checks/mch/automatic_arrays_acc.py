# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class AutomaticArraysCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-cce', 'PrgEnv-pgi']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True

        # This tets requires an MPI compiler, although it uses a single task
        self.num_tasks = 1
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcepath = 'automatic_arrays_OpenACC.F90'
        self.build_system = 'SingleSource'
        self.build_system.fflags = ['-O2']
        self.sanity_patterns = sn.assert_found(r'Result: ', self.stdout)
        self.perf_patterns = {
            'time': sn.extractsingle(r'Timing:\s+(?P<time>\S+)',
                                     self.stdout, 'time', float)
        }
        self.arrays_reference = {
            'PrgEnv-cray': {
                'daint:gpu': {'time': (5.7E-05, None, 0.15, 's')},
                'dom:gpu': {'time': (5.7E-05, None, 0.15, 's')},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'time': (7.5E-05, None, 0.15, 's')},
                'dom:gpu': {'time': (7.5e-05, None, 0.15, 's')},
            }
        }

        self.maintainers = ['AJ', 'MKr']
        self.tags = {'production', 'mch', 'craype'}

    @rfm.run_before('compile')
    def setflags(self):
        if self.current_environ.name.startswith('PrgEnv-cray'):
            envname = 'PrgEnv-cray'
            self.build_system.fflags += ['-hacc', '-hnoomp']
        elif self.current_environ.name.startswith('PrgEnv-cce'):
            envname = 'PrgEnv-cce'
            self.build_system.fflags += ['-hacc', '-hnoomp']
        elif self.current_environ.name.startswith('PrgEnv-pgi'):
            envname = 'PrgEnv-pgi'
            self.build_system.fflags += ['-acc']
            if self.current_system.name in ['arolla', 'tsa']:
                self.build_system.fflags += ['-ta=tesla,cc70']
            elif self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags += ['-ta=tesla,cc60', '-Mnorpath']
        else:
            envname = self.current_environ.name

        self.reference = self.arrays_reference[envname]

    @rfm.run_before('compile')
    def cdt2008_pgi_workaround(self):
        cdt = osext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt == '20.08'):
            self.variables.update({'CUDA_HOME': '$CUDATOOLKIT_HOME'})
