# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


class HPCGHookMixin(rfm.RegressionMixin):
    @run_before('run')
    def guide_node_guess(self):
        '''Guide the node guess based on the test's needs.'''

        # Prelim guess
        ntasks_per_node = self.num_tasks_per_node or 1
        self.job.num_tasks_per_node = ntasks_per_node
        n = int(self.job.guess_num_tasks()/ntasks_per_node)

        def first_factor(x):
            if x <= 1:
                return 1

            for i in range(2, x+1):
                if x % i == 0:
                    return i

        def is_prime(x):
            return first_factor(x) == x

        # Correct the prelim node numbers
        # n = 11 would be the first number not meeting HPCG's condition
        while n > 10:
            x = int(n/first_factor(n))

            # If x==1, n is prime
            # if x > 8 and prime, it would also not meet HPCG's aspect ratio
            if x == 1 or (is_prime(x) and x > 8):
                n -= 1
            else:
                break

        self.num_tasks = int(n*ntasks_per_node)
        self.num_tasks_per_node = ntasks_per_node


@rfm.simple_test
class HPCGCheckRef(rfm.RegressionTest, HPCGHookMixin):
    descr = 'HPCG reference benchmark'
    valid_systems = ['daint:mc', 'daint:gpu', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    build_system = 'Make'
    sourcesdir = 'https://github.com/hpcg-benchmark/hpcg.git'
    executable = 'bin/xhpcg'
    executable_opts = ['--nx=104', '--ny=104', '--nz=104', '-t2']
    # use glob to catch the output file suffix dependent on execution time
    output_file = sn.getitem(sn.glob('HPCG*.txt'), 0)
    num_tasks = 0
    num_cpus_per_task = 1

    reference = {
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

    maintainers = ['SK', 'EK']
    tags = {'diagnostic', 'benchmark', 'craype', 'external-resources'}

    @run_after('init')
    def set_modules(self):
        if self.current_system.name in {'daint', 'dom'}:
            self.modules = ['craype-hugepages8M']

    @run_before('compile')
    def set_build_opts(self):
        self.build_system.options = ['arch=MPI_GCC_OMP']


    @property
    @deferrable
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @run_before('compile')
    def set_tasks(self):
        if self.current_partition.processor.num_cores:
            self.num_tasks_per_node = (
                self.current_partition.processor.num_cores
            )
        else:
            self.num_tasks_per_node = 1

    @performance_function('Gflop/s')
    def gflops(self):
        num_nodes = self.num_tasks_assigned // self.num_tasks_per_node
        return (
            sn.extractsingle(
            r'HPCG result is VALID with a GFLOP\/s rating of=\s*'
            r'(?P<perf>\S+)',
            self.output_file, 'perf',  float) / num_nodes
        )

    @sanity_function
    def validate_passed(self):
        return sn.all([
            sn.assert_eq(4, sn.count(
                sn.findall(r'PASSED', self.output_file))),
            sn.assert_eq(0, self.num_tasks_assigned % self.num_tasks_per_node)
        ])


@rfm.simple_test
class HPCGCheckMKL(rfm.RegressionTest, HPCGHookMixin):
    descr = 'HPCG benchmark Intel MKL implementation'
    valid_systems = ['daint:mc', 'dom:mc', 'daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-intel']
    modules = ['craype-hugepages8M']
    build_system = 'Make'
    prebuild_cmds = ['cp -r ${MKLROOT}/benchmarks/hpcg/* .',
                     'mv Make.CrayXC setup', './configure CrayXC']

    num_tasks = 0
    problem_size = 104
    variables = {
        'HUGETLB_VERBOSE': '0',
        'MPICH_MAX_THREAD_SAFETY': 'multiple',
        'MPICH_USE_DMAPP_COLL': '1',
        'PMI_NO_FORK': '1',
        'KMP_AFFINITY': 'granularity=fine,compact'
    }

    executable = 'bin/xhpcg_avx2'
    reference = {
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

    maintainers = ['SK']
    tags = {'diagnostic', 'benchmark', 'craype'}

    @run_before('run')
    def set_exec_opt(self):
        self.executable_opts = [f'--nx={self.problem_size}',
                                f'--ny={self.problem_size}',
                                f'--nz={self.problem_size}', '-t2']

    @property
    @deferrable
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @property
    @deferrable
    def outfile_lazy(self):
        pattern = (f'n{self.problem_size}-{self.job.num_tasks}p-'
                   f'{self.num_cpus_per_task}t*.*')
        return sn.getitem(sn.glob(pattern), 0)

    @run_before('compile')
    def set_tasks(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 12
        else:
            self.num_tasks_per_node = 4
            self.num_cpus_per_task = 18

    @performance_function('Gflop/s')
    def gflops(self):
        # since this is a flexible test, we divide the extracted
        # performance by the number of nodes and compare
        # against a single reference
        num_nodes = self.num_tasks_assigned // self.num_tasks_per_node
        return (
            sn.extractsingle(
            r'HPCG result is VALID with a GFLOP\/s rating of(=|:)\s*'
            r'(?P<perf>\S+)',
            self.outfile_lazy, 'perf',  float) / num_nodes
        )

    @sanity_function
    def validate_passed(self):
        return sn.all([
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
class HPCG_GPUCheck(rfm.RunOnlyRegressionTest, HPCGHookMixin):
    descr = 'HPCG benchmark on GPUs'
    # there's no binary with support for CUDA 10 yet
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    modules = ['craype-accel-nvidia60', 'craype-hugepages8M']
    executable = 'xhpcg_gpu_3.1'
    num_tasks = 0
    num_tasks_per_node = 1
    output_file = sn.getitem(sn.glob('*.yaml'), 0)
    reference = {
        'daint:gpu': {
            'gflops': (94.7, -0.1, None, 'Gflop/s')
        },
        'dom:gpu': {
            'gflops': (94.7, -0.1, None, 'Gflop/s')
        },
    }
    maintainers = ['SK', 'VH']

    @run_after('setup')
    def set_num_tasks(self):
        if self.current_partition.processor.num_cores:
            self.num_cpus_per_task = (
                self.current_partition.processor.num_cores
            )
        else:
            self.skip(msg='number of cores is not set in the configuration')

    @run_after('init')
    def set_sourcedir(self):
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'HPCG')

    @run_after('init')
    def set_variables(self):
        self.variables = {
            'PMI_NO_FORK': '1',
            'MPICH_USE_DMAPP_COLL': '1',
            'OMP_SCHEDULE': 'static',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'HUGETLB_VERBOSE': '0',
            'HUGETLB_DEFAULT_PAGE_SIZE': '8M',
        }

    @run_before('run')
    def set_exec_permissions(self):
        self.prerun_cmds = ['chmod +x %s' % self.executable]

    @sanity_function
    def validate_passed(self):
        return sn.all([
            sn.assert_eq(4, sn.count(
                sn.findall(r'PASSED', self.output_file))),
            sn.assert_eq(0, self.num_tasks_assigned % self.num_tasks_per_node)
        ])

    @performance_function('Gflop/s')
    def gflops(self):
        num_nodes = self.num_tasks_assigned // self.num_tasks_per_node
        return (
            sn.extractsingle(
            r'HPCG result is VALID with a GFLOP\/s rating of:\s*'
            r'(?P<perf>\S+)',
            self.output_file, 'perf',  float) / num_nodes
        )

    @property
    @deferrable
    def num_tasks_assigned(self):
        return self.job.num_tasks
