# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext


class AffinityTestBase(rfm.RegressionTest):

    # FIXME: PR #1699 should introduce
    # var('cpu_bind', str)
    # var('hint', str)

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
        self.sanity_patterns = self.assert_consumed_cpu_set()
        self.maintainers = ['RS', 'SK']
        self.tags = {'production', 'scs', 'maintenance', 'craype'}

        # Dict with the partition's topology - output of "lscpu -e"
        self.topology = {
            'dom:gpu':   'topo_dom_gpu.txt',
            'dom:mc':    'topo_dom_mc.txt',
            'daint:gpu': 'topo_dom_gpu.txt',
            'daint:mc':  'topo_dom_mc.txt',
            'eiger:mc':  'topo_eiger_mc.txt',
        }

    @rfm.run_before('compile')
    def set_topo_file(self):
        cp = self.current_partition.fullname
        self.topo_file = self.topology[cp]

    @rfm.run_after('compile')
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

        This hook requires the reference file (self.topo_file), so the earlies
        it can run is after the compilation stage, once the required files have
        been copied over to the stage directory.
        '''

        with osext.change_dir(self.stagedir):
            lscpu = sn.extractall(
                r'^\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)',
                self.topo_file, 0,
                lambda x: [int(xi) for xi in x.split()]
            ).evaluate()

        # Build the cpu set
        self.cpu_set = set(map(lambda x: x[0], lscpu))
        self.num_cpus = len(self.cpu_set)

        # Build the numa sets
        self.num_numa_nodes = len(set(map(lambda x: x[1], lscpu)))
        self.num_cpus_per_core = int(
            self.num_cpus/len(set(map(lambda x: x[3], lscpu)))
        )
        self.numa_nodes = {}
        for i in range(self.num_numa_nodes):
            self.numa_nodes[i] = set(
                map(lambda y: y[0], filter(lambda x: x[1]==i, lscpu))
            )

        # Build the socket sets
        self.num_sockets = len(set(map(lambda x: x[2], lscpu)))
        self.sockets = {}
        for i in range(self.num_sockets):
            self.sockets[i] = set(
                map(lambda y: y[0], filter(lambda x: x[2]==i, lscpu))
            )

        # Store the lscpu output
        self.__lscpu = lscpu

    def get_sibiling_cpus(self, cpuid, by=None):
        '''Return a cpu set where cpuid belongs to.

        The cpu set can be extracted by matching core, numa domain or socket.
        This is controlled by the `by` argument.
        '''
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
    def assert_consumed_cpu_set(self):
        '''Check that all the resources have been consumed.'''
        return sn.assert_eq(self.cpu_set, set())

    @rfm.run_after('run')
    def parse_output(self):

        re_aff_cpus = r'CPU affinity: \[\s+(?P<cpus>[\d+\s+]+)\]'
        re_aff_thrds = r'^Tag:[^\n\r]*Thread:\s+(?P<thread>\d+)'
        re_aff_ranks = r'^Tag:[^\n\r]*Rank:\s+(?P<rank>\d+)[\s+\S+]'
        def parse_cpus(x):
            return sorted([int(xi) for xi in x.split()])

        with osext.change_dir(self.stagedir):
            self.aff_cpus = sn.extractall(
                re_aff_cpus, self.stdout, 'cpus', parse_cpus
            ).evaluate()

            self.aff_thrds = sn.extractall(
                re_aff_thrds, self.stdout, 'thread', int
            )

            self.aff_ranks = sn.extractall(
                re_aff_ranks, self.stdout, 'rank', int
            )

    @rfm.run_before('run')
    def set_multithreading(self):
        cp = self.current_partition.fullname
        mthread = self.system.get(cp, {}).get('multithreading', None)
        if mthread:
            self.use_multithreading = mthread

    @rfm.run_before('run')
    def set_launcher(self):
        cp = self.current_partition.fullname
        cpu_bind = self.system.get(cp, {}).get('cpu-bind', None)
        if cpu_bind:
            self.job.launcher.options += [f'--cpu-bind={cpu_bind}']
        elif hasattr(self, 'cpu_bind'):
            self.job.launcher.options += [f'--cpu-bind={self.cpu_bind}']

        hint = self.system.get(cp, {}).get('hint', None)
        if hint:
            self.job.launcher.options += [f'--hint={hint}']

        elif hasattr(self, 'hint'):
            self.job.launcher.options += [f'--hint={self.hint}']

class AffinityOpenMPBase(AffinityTestBase):
    '''Extend affinity base with OMP hooks.

    The tests derived from this class book the full node and place the
    threads acordingly based exclusively on the OMP_BIND env var. The
    number of total OMP_THREADS will vary depending on what are we
    binding the OMP threads to (e.g. if we bind to sockets, we'll have as
    many threads as sockets).
    '''

    # FIXME: PR #1699 should make these vars instead of parameters.
    parameter('omp_bind')
    parameter('omp_proc_bind', ['spread'])

    def __init__(self):
        super().__init__()
        self.num_tasks = 1

    @property
    def _num_cpus_per_task(self):
        return self.num_cpus

    @rfm.run_before('run')
    def set_num_cpus_per_task(self):
        self.num_cpus_per_task = self._num_cpus_per_task

    @rfm.run_before('run')
    def set_omp_vars(self):
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_omp_threads),
            'OMP_PLACES': self.omp_bind,
            'OMP_PROC_BIND': self.omp_proc_bind,
        }

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        raise ValueError('this function must be overridden')


@rfm.simple_test
class PinToCPUs_OMP_FullNode(AffinityOpenMPBase):
    '''Full node booked with one OMP thread per cpu.'''
    parameter('omp_bind', ['threads'])

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per CPU.'
        self.system = {
            # System-dependent settings here
        }

    @property
    def num_omp_threads(self):

        # One OMP thread per cpu
        return self.num_cpus

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to cpus.'''
        for cpus_bound_to_thread in self.aff_cpus:
            if ((len(cpus_bound_to_thread) > 1) or
                not all(x in self.cpu_set for x in cpus_bound_to_thread)):

                # This will force the sanity function to fail.
                self.cpu_set.update([-1])

            self.cpu_set -= set(cpus_bound_to_thread)


@rfm.simple_test
class PinToCores_OMP_FullNode(AffinityOpenMPBase):
    '''Full node booked with one OMP thread per core.'''
    parameter('omp_bind', ['cores'])

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per core.'
        self.system = {
            # System-dependent settings here
        }

    @property
    def num_omp_threads(self):

        # One OMP thread per core
        return int(self.num_cpus/self.num_cpus_per_core)

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to cores.'''
        for cpus_bound_to_thread in self.aff_cpus:

            # Get CPU sibilings by core
            cpu_sibilings = self.get_sibiling_cpus(cpus_bound_to_thread[0], by='core')

            # If there is more than 1 CPU, it must belong to the same core
            if (not all(x in self.cpu_set for x in cpus_bound_to_thread) or
                not all(x in cpu_sibilings for x in cpus_bound_to_thread)):

                # This will force the sanity function to fail.
                self.cpu_set.update([-1])

            self.cpu_set -= cpu_sibilings


@rfm.simple_test
class PinToSockets_OMP_FullNode(AffinityOpenMPBase):
    '''Full node booked with one OMP thread per socket.'''
    parameter('omp_bind', ['socket'])

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per socket.'
        self.system = {
            # System-dependent settings here
        }

    @property
    def num_omp_threads(self):

        # One OMP thread per core
        return self.num_sockets

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to sockets.'''
        for cpus_bound_to_thread in self.aff_cpus:

            # Get CPU sibilings by core
            cpu_sibilings = self.get_sibiling_cpus(cpus_bound_to_thread[0], by='socket')

            # If there is more than 1 CPU, it must belong to the same socket
            if (not all(x in self.cpu_set for x in cpus_bound_to_thread) or
                not all(x in cpu_sibilings for x in cpus_bound_to_thread)):

                # This will force the sanity function to fail.
                self.cpu_set.update([-1])

            self.cpu_set -= cpu_sibilings


@rfm.simple_test
class PinToCores_OMP_nomultithread(PinToCores_OMP_FullNode):
    '''Only one cpu per core booked.'''
    parameter('omp_bind', ['cores'])

    # FIXME: PR #1699
    # set_var('hint', 'nomultithread')

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per core.'

        # System independent settings
        self.hint = 'nomultithread'

        self.system = {
            # System-dependent settings here
        }

    @property
    def _num_cpus_per_task(self):
        return self.num_omp_threads


@rfm.simple_test
class OneTaskPerSocket_OMP_nomultithread(AffinityOpenMPBase):
    '''Only one cpu per socket booked.'''
    parameter('omp_bind', ['sockets'])
    parameter('omp_proc_bind', ['close'])

    # FIXME: PR #1699
    # set_var('hint', 'nomultithread')

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per core.'

        # System independent settings
        self.num_tasks = 2
        self.use_multithreading = False

        self.system = {
            # System-dependent settings here
        }

    @property
    def num_omp_threads(self):
        return int(self.num_cpus/self.num_cpus_per_core/self.num_sockets)

    @property
    def _num_cpus_per_task(self):
        return self.num_omp_threads

    @rfm.run_before('sanity')
    def consume_cpu_set(self):

        threads_in_socket = [0]*self.num_sockets
        def get_socket_id(cpuid):
            for i in range(self.num_sockets):
                if cpuid in self.sockets[i]:
                    return i

        for cpus_bound_to_thread in self.aff_cpus:

            # Count the number of OMP threads that live on each socket
            threads_in_socket[get_socket_id(cpus_bound_to_thread[0])] += 1

            # Get CPU sibilings by socket
            cpu_sibilings = self.get_sibiling_cpus(cpus_bound_to_thread[0], by='socket')

            # If there is more than 1 CPU, it must belong to the same socket
            if ((not self.num_omp_threads == len(cpus_bound_to_thread)) or
                not all(x in cpu_sibilings for x in cpus_bound_to_thread)):

                # This will force the sanity function to fail.
                self.cpu_set.update([-1])

        # Remove the sockets the cpu set.
        for i, socket in enumerate(self.sockets.values()):
            if threads_in_socket[i] == self.num_omp_threads:
                self.cpu_set -= socket


@rfm.simple_test
class OneTaskPerSocket_OMP(OneTaskPerSocket_OMP_nomultithread):
    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per core.'

        # System independent settings
        self.num_tasks = 2
        self.use_multithreading = True
        self.system = {
            # System-dependent settings here
        }

    @property
    def num_omp_threads(self):
        return int(self.num_cpus/self.num_sockets)
