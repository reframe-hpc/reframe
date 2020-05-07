# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class HPCGCheckRef(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'HPCG reference benchmark'
        self.valid_systems = ['daint:mc', 'daint:gpu', 'dom:gpu', 'dom:mc',
                              'tiger:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in {'daint', 'dom'}:
            self.modules = ['craype-hugepages8M']

        self.build_system = 'Make'
        self.build_system.options = ['arch=MPI_GCC_OMP']
        self.sourcesdir = 'https://github.com/hpcg-benchmark/hpcg.git'

        # FIXME: Remove this after the OpenMP pragma gets fixed in hpcg master
        self.prebuild_cmd = [
            'git checkout 9484cd7f2c4744c783abbdcfd4f5cc34807b42b1'
        ]
        self.executable = 'bin/xhpcg'
        self.executable_opts = ['--nx=104', '--ny=104', '--nz=104', '-t2']
        # use glob to catch the output file suffix dependent on execution time
        self.output_file = sn.getitem(sn.glob('HPCG*.txt'), 0)

        self.num_tasks = 0
        self.num_cpus_per_task = 1
        self.system_num_tasks = {
            'daint:mc':  36,
            'daint:gpu': 12,
            'dom:mc':  36,
            'dom:gpu': 12
        }

        self.reference = {
            'daint:gpu': {
                'gflops': (7.6, -0.1, None, 'Gflop/s')
            },
            'daint:mc': {
                'gflops': (13.4, -0.1, None, 'Gflop/s')
            },
            'dom:gpu': {
                'gflops': (7.6, -0.1, None, 'Gflop/s')
            },
            'dom:mc': {
                'gflops': (13.4, -0.1, None, 'Gflop/s')
            }
        }

        self.maintainers = ['SK', 'EK']
        self.tags = {'diagnostic', 'benchmark', 'craype', 'external-resources'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @rfm.run_before('compile')
    def set_tasks(self):
        self.num_tasks_per_node = self.system_num_tasks.get(
            self.current_partition.fullname, 1
        )

    @rfm.run_before('performance')
    def set_performance(self):
        num_nodes = self.num_tasks_assigned / self.num_tasks_per_node
        self.perf_patterns = {
            'gflops': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of=\s*'
                r'(?P<perf>\S+)',
                self.output_file, 'perf',  float) / num_nodes
        }

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.all([
            sn.assert_eq(4, sn.count(
                sn.findall(r'PASSED', self.output_file))),
            sn.assert_eq(0, self.num_tasks_assigned % self.num_tasks_per_node)
        ])


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class HPCGCheckMKL(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'HPCG benchmark Intel MKL implementation'
        self.valid_systems = ['daint:mc', 'dom:mc', 'daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['craype-hugepages8M']
        self.build_system = 'Make'
        self.prebuild_cmd = ['cp -r ${MKLROOT}/benchmarks/hpcg/* .',
                             'mv Make.CrayXC setup',
                             './configure CrayXC']

        self.num_tasks = 0
        self.num_tasks_per_core = 2
        self.problem_size = 104

        self.variables = {
            'HUGETLB_VERBOSE': '0',
            'MPICH_MAX_THREAD_SAFETY': 'multiple',
            'MPICH_USE_DMAPP_COLL': '1',
            'PMI_NO_FORK': '1',
            'KMP_HW_SUBSET': '9c,2t',
            'KMP_AFFINITY': 'granularity=fine,compact'
        }

        self.executable = 'bin/xhpcg_avx2'
        self.executable_opts = ['--nx=%d' % self.problem_size,
                                '--ny=%d' % self.problem_size,
                                '--nz=%d' % self.problem_size, '-t2']

        self.reference = {
            'dom:mc': {
                'gflops': (22, -0.1, None, 'Gflop/s')
            },
            'daint:mc': {
                'gflops': (22, -0.1, None, 'Gflop/s')
            },
            'dom:gpu': {
                'gflops': (10.7, -0.1, None, 'Gflop/s')
            },
            'daint:gpu': {
                'gflops': (10.7, -0.1, None, 'Gflop/s')
            },
        }

        self.maintainers = ['SK']
        self.tags = {'diagnostic', 'benchmark', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @property
    @sn.sanity_function
    def outfile_lazy(self):
        pattern = 'n%d-%dp-%dt-*.yaml' % (self.problem_size,
                                          self.job.num_tasks,
                                          self.num_cpus_per_task)
        return sn.getitem(sn.glob(pattern), 0)

    @rfm.run_before('compile')
    def set_tasks(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 12
        else:
            self.num_tasks_per_node = 4
            self.num_cpus_per_task = 18

    @rfm.run_before('performance')
    def set_performance(self):
        # since this is a flexible test, we divide the extracted
        # performance by the number of nodes and compare
        # against a single reference
        num_nodes = self.num_tasks_assigned / self.num_tasks_per_node
        self.perf_patterns = {
            'gflops': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of:\s*'
                r'(?P<perf>\S+)',
                self.outfile_lazy, 'perf',  float) / num_nodes
        }

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.all([
            sn.assert_not_found(
                r'invalid because the ratio',
                self.outfile_lazy,
                msg='number of processes assigned could not be factorized'
            ),
            sn.assert_eq(
                4, sn.count(sn.findall(r'PASSED', self.outfile_lazy))
            ),
            sn.assert_eq(0, self.num_tasks_assigned % self.num_tasks_per_node)
        ])


@rfm.simple_test
class HPCG_GPUCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.maintainers = ['SK', 'VH']
        self.descr = 'HPCG benchmark on GPUs'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'HPCG')

        # there's no binary with support for CUDA 10 yet
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['craype-accel-nvidia60', 'craype-hugepages8M']
        self.executable = 'xhpcg_gpu_3.1'
        self.pre_run = ['chmod +x %s' % self.executable]
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12
        self.variables  = {
            'PMI_NO_FORK': '1',
            'MPICH_USE_DMAPP_COLL': '1',
            'OMP_SCHEDULE': 'static',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'HUGETLB_VERBOSE': '0',
            'HUGETLB_DEFAULT_PAGE_SIZE': '8M',
        }

        self.output_file = sn.getitem(sn.glob('*.yaml'), 0)

        self.reference = {
            'daint:gpu': {
                'gflops': (94.7, -0.1, None, 'Gflop/s')
            },
            'dom:gpu': {
                'gflops': (94.7, -0.1, None, 'Gflop/s')
            },
        }

        num_nodes = self.num_tasks_assigned / self.num_tasks_per_node
        self.perf_patterns = {
            'gflops': sn.extractsingle(
                r'HPCG result is VALID with a GFLOP\/s rating of:\s*'
                r'(?P<perf>\S+)',
                self.output_file, 'perf',  float) / num_nodes
        }

        self.sanity_patterns = sn.all([
            sn.assert_eq(4, sn.count(
                sn.findall(r'PASSED', self.output_file))),
            sn.assert_eq(0, self.num_tasks_assigned % self.num_tasks_per_node)
        ])

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
