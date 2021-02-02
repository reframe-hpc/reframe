# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
import copy

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext

class ProcTopo:
    '''Contains information on the processor's topology.

    This processor information is filled after caling the collect
    member function. This method collects:
      - num_cpus
      - num_threads_per_core
      - num_sockets
      - num_numa_nodes
      - thread_stride: CPU ID stride for threads using the same core.
      - cpu_set: Processor CPU set.
      - numa_nodes: Dictionary containing the CPU sets for each NUMA node.
      - sockets: Dictionary containing the CPU sets for each socket.
    '''

    def collect(self):
        '''Retrieves the processor topology and builds some handy CPU sets.'''
        self.lscpu()
        self.thread_stride = self._thread_stride(0)
        self._sockets = self._socket_sets()

    def thread_sibilings(self, cpuid):
        '''Return the CPU set for the same core as cpuid.'''
        while (cpuid - self.thread_stride) > 0:
            cpuid -= self.thread_stride

        return set(range(cpuid, self.num_cpus, self.thread_stride))

    def _socket_sets(self):
        '''Return a dict with the CPU sets from each socket.'''
        cpuset = set(range(0, self.num_cpus-1))
        setcount = 0
        socket_sets = {}
        while cpuset != set():
            socket_sets[setcount] = self.core_sibilings(next(iter(cpuset)))
            cpuset -= socket_sets[setcount]
            setcount += 1

        if setcount != self.num_sockets:
            raise ValueError('error building the socket CPU sets')

        return socket_sets

    def core_sibilings(self, cpuid):
         '''Return the CPU set for the same socket as cupid.'''
         core_sibilings = osext.run_command(
             f'cat /sys/devices/system/cpu/cpu{cpuid}'
             f'/topology/core_siblings_list',
             log = False
         ).stdout.split(',')

         cpu_socket_set = set()
         for item in core_sibilings:
             subset = [int(x) for x in item.split('-')]
             if len(subset) > 1:
                 cpu_socket_set.update(range(subset[0], subset[1]+1))
             else:
                 cpu_socket_set.update([subset[0]])

         return cpu_socket_set

    def _thread_stride(self, cpuid):
        '''Get the cpu numbering stride for threads using the same core.'''
        thread_sibilings = [int(x) for x in osext.run_command(
                f'cat /sys/devices/system/cpu/cpu{cpuid}'
                f'/topology/thread_siblings_list',
                log = False
            ).stdout.split(',')
        ]

        if len(thread_sibilings) > 1:
            return thread_sibilings[1] - thread_sibilings[0]
        else:
            return None

    def lscpu(self):
        '''Fetch the basic proc info from the lscpu command.

        This will work out the number of CPUs, the number of CPUs per core, the
        number of sockets, number of cores per socket, number of numa nodes,
        and a dict containing the CPU sets for each numa node.
        '''
        import re
        lscpu_out = osext.run_command('lscpu', log=False).stdout
        def fetch_field(patt, conv):
            return conv(re.findall(patt, lscpu_out, re.MULTILINE)[0])

        self.num_cpus = fetch_field('^\s*CPU\(s\):\s*([\d]+)', int)
        cpu_range = fetch_field(
            '^\s*On-line CPU\(s\) list:\s*([\d]+-[\d]+)',
            lambda x: [int(y) for y in x.split('-')]
        )
        self._cpu_set = set(range(cpu_range[0], cpu_range[1]+1))
        self.num_threads_per_core = fetch_field(
            '^\s*Thread\(s\) per core:\s*([\d]+)', int
        )
        self.num_sockets = fetch_field('^\s*Socket\(s\):\s*([\d]+)', int)
        self.cores_per_socket = fetch_field(
            '^\s*Core\(s\) per socket:\s*([\d]+)', int
        )
        self.num_numa_nodes = fetch_field('^\s*NUMA node\(s\):\s*([\d]+)', int)

        # Fetch the ranges in the NUMA nodes
        self._numa_nodes = {}
        for i in range(self.num_numa_nodes):
            self._numa_nodes[i] = set()
            numa_node = fetch_field(
                r'^\s*NUMA node%d CPU\(s\):\s*([\d\-,]+)' % i,
                lambda x: x.split(',')
            )

            for j in numa_node:
                numa_range = [int(x) for x in j.split('-')]
                if len(numa_range) > 1:
                    self._numa_nodes[i].update(set(range(numa_range[0], numa_range[1]+1)))
                else:
                    self._numa_nodes[i].update(set([numa_range[0]]))

        if len(self._numa_nodes) != self.num_numa_nodes:
            raise ValueError('error building the NUMA CPU sets')

    @property
    def cpu_set(self):
        return copy.deepcopy(self._cpu_set)

    @property
    def numa_nodes(self):
        return copy.deepcopy(self._numa_nodes)

    @property
    def sockets(self):
        return copy.deepcopy(self._sockets)


class AffinityTestBase(rfm.RegressionTest):
    parameter('variant')

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc', 'eiger:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.build_system = 'Make'
        self.build_system.options = ['-C affinity', 'MPI=1']
        # The github URL can not be specifid as `self.sourcedir` as that
        # would prevent the src folder from being copied to stage which is
        # necessary since these tests need files from it.
        self.sourcesdir = os.path.join('src/affinity_ref')
        self.prebuild_cmds = ['git clone https://github.com/vkarak/affinity']
        self.executable = './affinity/affinity'

        # Processor topology
        self.topology = ProcTopo()

#        self.sanity_patterns = self.set_sanity()
        self.maintainers = ['RS', 'SK']
        self.tags = {'production', 'scs', 'maintenance', 'craype'}

    @rfm.run_after('setup')
    def get_topology(self):
        '''Populate the structure with the processor's topology'''
        self.topology.collect()
        self.cpu_set = self.topology.cpu_set

    def pop_cpu(self, *cpu_id):
        '''Remove the cpu_id from the CPU set - only if present.'''
        print('+++++++++++++', cpu_id)
        sn.assert_eq(
            list(filter(lambda x: x not in self.cpu_set, cpu_id)),
            []
        )
        print('*************', set(cpu_id))
        self.cpu_set -= set(cpu_id)

    def pop_core(self, *cpu_id):
        '''Remove the cpus sharing cpu_id's core from the CPU set.'''
        for cpu in cpu_id:
            thread_sibilings = self.topology.thread_sibilings(cpu)

            # Make sure all these thread sibilings are still in the cpu set
            sn.assert_eq(
                list(filter(
                    lambda x: x not in self.cpu_set,
                    thread_sibilings)
                ), []
            )
            self.cpu_set -= set(thread_sibilings)

    def pop_socket(self, *cpu_id):
        '''Remove the cpus sharing cpu_id's socket from the CPU set.'''
        for cpu in cpu_id:
            core_sibilings = self.topology.core_sibilings(cpu)

            # Make sure all these core sibilings are still in the cpu set
            sn.assert_eq(
                list(filter(
                    lambda x: x not in self.cpu_set,
                    core_sibilings)
                ), []
            )
            self.cpu_set -= set(core_sibilings)

    @property
    def cpus_with_affinity(self):
        def parse_cpus(x):
            return [int(xi) for xi in x.split()]

        return sn.extractall(
            r'CPU affinity: \[\s+(?P<cpus>[\d+\s+]+)\]',
            self.stdout, 'cpus', parse_cpus
        )

    @property
    def threads(self):
        return sn.extractall(
            r'^Tag:[^\n\r]*Thread:\s+(?P<thread>\d+)',
            self.stdout, 'thread', int
        )

    @property
    def ranks(self):
        return sn.extractall(
            r'^Tag:[^\n\r]*Rank:\s+(?P<rank>\d+)[\s+\S+]',
            'rank', int
        )

##    @sn.sanity_function
##    def set_sanity(self):
##        def parse_cpus(x):
##            return sorted([int(xi) for xi in x.split()])
##
##        re_aff_cores = r'CPU affinity: \[\s+(?P<cpus>[\d+\s+]+)\]'
##        self.aff_cores = sn.extractall(
##            re_aff_cores, self.stdout, 'cpus', parse_cpus)
##        #print('####', self.aff_cores)
##        ref_key = 'ref_' + self.current_partition.fullname
##        self.ref_cores = sn.extractall(
##            re_aff_cores, self.cases[self.variant][ref_key],
##            'cpus', parse_cpus)
##        re_aff_thrds = r'^Tag:[^\n\r]*Thread:\s+(?P<thread>\d+)'
##        self.aff_thrds = sn.extractall(re_aff_thrds, self.stdout, 'thread',
##                                       int)
##        #print('####', self.aff_thrds)
##        self.ref_thrds = sn.extractall(
##            re_aff_thrds, self.cases[self.variant][ref_key],
##            'thread', int)
##        re_aff_ranks = r'^Tag:[^\n\r]*Rank:\s+(?P<rank>\d+)[\s+\S+]'
##        self.aff_ranks = sn.extractall(re_aff_ranks, self.stdout, 'rank', int)
##        #print('####', self.aff_ranks)
##        self.ref_ranks = sn.extractall(
##            re_aff_ranks, self.cases[self.variant][ref_key],
##            'rank', int)
##
##        # Ranks and threads can be extracted into lists in order to compare
##        # them since the affinity programm prints them in ascending order.
##        return sn.all([
##            sn.assert_eq(self.aff_thrds, self.ref_thrds),
##            sn.assert_eq(self.aff_ranks, self.ref_ranks),
##            sn.assert_eq(sn.sorted(self.aff_cores), sn.sorted(self.ref_cores))
##        ])

    @rfm.run_before('run')
    def set_multithreading(self):
        partname = self.current_partition.fullname
        self.use_multithreading = self.cases[partname]['multithreading']


@rfm.simple_test
class AffinityBindThreadsOMPTest(AffinityTestBase):
    parameter('variant', ['omp_bind_threads'])

    def __init__(self):
        super().__init__()
        self.descr = 'Checking OMP threads are pinned to cpus'
        self.sanity_patterns = self.using_all_cpus()
        self.cases = {
            'dom:gpu': {
                'multithreading': None,
                'OMP_PLACES': 'threads',
            },
        }

    @rfm.run_before('run')
    def set_num_cpus_per_task(self):
        self.num_cpus_per_task = self.topology.num_cpus

    @rfm.run_before('run')
    def set_tasks_per_core(self):
        partname = self.current_partition.fullname
        self.num_tasks = 1
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PLACES': self.cases[partname]['OMP_PLACES']
            # OMP_PROC_BIND is set to TRUE if OMP_PLACES is defined.
            # Both OMP_PROC_BIND values CLOSE and SPREAD give the same
            # result as OMP_PROC_BIND=TRUE when all cores are requested.
        }

    @sn.sanity_function
    def using_all_cpus(self):
       print("$$$$", self.cpus_with_affinity)
       for cpus in self.cpus_with_affinity:
           print("  %% ", cpus)
           self.pop_cpu(*cpus)
           print("~~", self.cpu_set)

       return sn.assert_eq(self.cpu_set, set())
