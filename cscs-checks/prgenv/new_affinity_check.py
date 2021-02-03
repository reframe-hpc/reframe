# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext


class AffinityTestBase(rfm.RegressionTest):
    parameter('variant')

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.build_system = 'Make'
        self.build_system.options = ['-C affinity', 'MPI=1']
        # The github URL can not be specifid as `self.sourcedir` as that
        # would prevent the src folder from being copied to stage which is
        # necessary since these tests need files from it.
        self.sourcesdir = os.path.join('src/affinity_ref')
        self.prebuild_cmds = ['git clone https://github.com/vkarak/affinity']
        self.executable = './affinity/affinity'
        self.sanity_patterns = self.get_num_cpus()
        self.maintainers = ['RS', 'SK']
        self.tags = {'production', 'scs', 'maintenance', 'craype'}

    @rfm.run_before('run')
    def read_proc_topo(self):
        '''Import the processor's topology from the reference file.

        This hook inserts the following attributes based on the processor's
        topology:
            - cpu_set: set containing all the cpu IDs
            - num_cpus
            - num_cpus_per_core
            - num_numa_nodes
            - num_sockets
            - numa_nodes: dictionary containing the cpu sets for each numa
                node. The keys of the dictionary are simply the ID of the
                numa node.
            - sockets: dictionary containing the cpu sets for each socket.
                The keys of the dictionary are simply the socket IDs.
        '''

        with osext.change_dir(self.stagedir):
            lscpu = sn.extractall(
                r'^\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)',
                self.ref_file, 0,
                lambda x: [int(xi) for xi in x.split()]
            ).evaluate()

        # Build the cpu set
        self.cpu_set = set(map(lambda x: x[0], lscpu))
        self.num_cpus = len(self.cpu_set)

        # Build the numa sets
        self.num_numa_nodes = len(set(map(lambda x: x[1], lscpu)))
        self.num_sockets = len(set(map(lambda x: x[2], lscpu)))
        self.num_cpus_per_core = int(
            len(set(map(lambda x: x[3], lscpu)))/self.num_cpus
        )
        self.numa_nodes = {}
        for i in range(self.num_numa_nodes):
            self.numa_nodes[i] = set(
                map(lambda y: y[0], filter(lambda x: x[1]==i, lscpu))
            )

        self.sockets = {}
        for i in range(self.num_sockets):
            self.sockets[i] = set(
                map(lambda y: y[0], filter(lambda x: x[2]==i, lscpu))
            )

        # Store the lscpu output
        self.__lscpu = lscpu

    def get_sibiling_cpus(self, cpuid, by=None):
        '''Return a cpu set of matching sibilings'''
        _map = {
            'core': 3,
            'socket': 2,
            'numa': 1,
        }

        if by is None:
            raise ValueError('must specify the sibiling level')
        else:
            try:
                sibiling_level = _map[by]
            except KeyError:
                raise ValueError('invalid sibiling level')

        sibiling_id = list(
            filter(lambda x: x[0]==cpuid, self.__lscpu)
        )[0][sibiling_level]

        return set(
            map(
                lambda y: y[0],
                filter(
                    lambda x: x[sibiling_level]==sibiling_id,
                    self.__lscpu
                )
            )
        )

    @sn.sanity_function
    def get_num_cpus(self):
        print(self.get_sibiling_cpus(44, by='core'))
        print(self.get_sibiling_cpus(44, by='socket'))
        return sn.assert_eq(len(self.cpu_set), 72)


    @rfm.run_before('sanity')
    def set_sanity(self):

        def parse_cpus(x):
            return sorted([int(xi) for xi in x.split()])

        re_aff_cores = r'CPU affinity: \[\s+(?P<cpus>[\d+\s+]+)\]'
        self.aff_cores = sn.extractall(
            re_aff_cores, self.stdout, 'cpus', parse_cpus)
        ref_key = 'ref_' + self.current_partition.fullname
        self.ref_cores = sn.extractall(
            re_aff_cores, self.cases[self.variant][ref_key],
            'cpus', parse_cpus)
        re_aff_thrds = r'^Tag:[^\n\r]*Thread:\s+(?P<thread>\d+)'
        self.aff_thrds = sn.extractall(re_aff_thrds, self.stdout, 'thread',
                                       int)
        self.ref_thrds = sn.extractall(
            re_aff_thrds, self.cases[self.variant][ref_key],
            'thread', int)
        re_aff_ranks = r'^Tag:[^\n\r]*Rank:\s+(?P<rank>\d+)[\s+\S+]'
        self.aff_ranks = sn.extractall(re_aff_ranks, self.stdout, 'rank', int)
        self.ref_ranks = sn.extractall(
            re_aff_ranks, self.cases[self.variant][ref_key],
            'rank', int)

##        # Ranks and threads can be extracted into lists in order to compare
##        # them since the affinity programm prints them in ascending order.
##        self.sanity_patterns = sn.all([
##            sn.assert_eq(self.aff_thrds, self.ref_thrds),
##            sn.assert_eq(self.aff_ranks, self.ref_ranks),
##            sn.assert_eq(sn.sorted(self.aff_cores), sn.sorted(self.ref_cores))
##        ])

    @rfm.run_before('run')
    def set_multithreading(self):
        self.use_multithreading = self.cases[self.variant]['multithreading']


@rfm.simple_test
class AffinityOpenMPTest(AffinityTestBase):
    parameter('variant', ['omp_bind_threads'])

    def __init__(self):
        super().__init__()
        self.descr = 'Checking the cpu affinity for OMP threads'

        self.ref_file = 'topo_dom_mc.txt'

        self.cases = {
            'omp_bind_threads': {
                'ref_daint:gpu': 'gpu_omp_bind_threads.txt',
                'ref_dom:gpu': 'gpu_omp_bind_threads.txt',
                'ref_daint:mc': 'mc_omp_bind_threads.txt',
                'ref_dom:mc': 'mc_omp_bind_threads.txt',
                'num_cpus_per_task:gpu': 24,
                'num_cpus_per_task:mc': 72,
                'ntasks_per_core': 2,
                'multithreading': None,
                'OMP_PLACES': 'threads',
            },
            'omp_bind_threads_nomultithread': {
                'ref_daint:gpu': 'gpu_omp_bind_threads_nomultithread.txt',
                'ref_dom:gpu': 'gpu_omp_bind_threads_nomultithread.txt',
                'ref_daint:mc': 'mc_omp_bind_threads_nomultithread.txt',
                'ref_dom:mc': 'mc_omp_bind_threads_nomultithread.txt',
                'num_cpus_per_task:gpu': 12,
                'num_cpus_per_task:mc': 36,
                'ntasks_per_core': None,
                # When `--hint=nomultithread` is not explicitly expecified only
                # half of the physical cores are used.
                'multithreading': False,
                'OMP_PLACES': 'threads',
            },
            'omp_bind_cores': {
                'ref_daint:gpu': 'gpu_omp_bind_cores.txt',
                'ref_dom:gpu': 'gpu_omp_bind_cores.txt',
                'ref_daint:mc': 'mc_omp_bind_cores.txt',
                'ref_dom:mc': 'mc_omp_bind_cores.txt',
                'num_cpus_per_task:gpu': 12,
                'num_cpus_per_task:mc': 36,
                'ntasks_per_core': 1,
                'multithreading': None,
                'OMP_PLACES': 'cores',
            },
        }

    @rfm.run_before('run')
    def set_tasks_per_core(self):
        partname = self.current_partition.name
        self.num_cpus_per_task = (
            self.cases[self.variant]['num_cpus_per_task:%s' % partname])
        if self.cases[self.variant]['ntasks_per_core']:
            self.num_tasks_per_core = (
                self.cases[self.variant]['ntasks_per_core'])

        self.num_tasks = 1
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PLACES': self.cases[self.variant]['OMP_PLACES']
            # OMP_PROC_BIND is set to TRUE if OMP_PLACES is defined.
            # Both OMP_PROC_BIND values CLOSE and SPREAD give the same
            # result as OMP_PROC_BIND=TRUE when all cores are requested.
        }


