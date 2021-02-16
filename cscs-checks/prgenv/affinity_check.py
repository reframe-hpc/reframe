# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
import json

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.exceptions import SanityError


class AffinityTestBase(rfm.RegressionTest):
    '''Base class for the affinity checks.

    It reads a reference file for each valid system, which allows this base
    class to figure out the processor's topology. The content of this reference
    file is simply the output of the `lscpu -e` command. For more info on this,
    see the `read_proc_topo` hook.
    '''

    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc', 'eiger:mc',
                              'ault:amdv100']
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
            'dom:gpu':    'topo_dom_gpu.json',
            'dom:mc':     'topo_dom_mc.json',
            'daint:gpu':  'topo_dom_gpu.json',
            'daint:mc':   'topo_dom_mc.json',
            'eiger:mc':   'topo_eiger_mc.json',
            'ault:amdv100': 'topo_ault_amdv100.json',
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

        This hook requires the reference file, so the earliest it can run is
        after the compilation stage, once the required files have been copied
        over to the stage directory.
        '''
        cp = self.current_partition.fullname
        with osext.change_dir(self.stagedir):
            with open(self.topology[cp], 'r') as topo:
                lscpu = json.load(topo)['cpus']

        # Build the cpu set
        self.cpu_set = set(map(lambda x: int(x['cpu']), lscpu))
        self.num_cpus = len(self.cpu_set)

        # Build the numa sets
        self.num_numa_nodes = len(set(map(lambda x: int(x['node']), lscpu)))
        self.num_cpus_per_core = int(
            self.num_cpus/len(set(map(lambda x: int(x['core']), lscpu)))
        )
        self.numa_nodes = {}
        for i in range(self.num_numa_nodes):
            self.numa_nodes[i] = set(
                map(lambda y: int(y['cpu']),
                    filter(lambda x: int(x['node']) == i, lscpu)
                    )
            )

        # Build the socket sets
        self.num_sockets = len(set(map(lambda x: int(x['socket']), lscpu)))
        self.sockets = {}
        for i in range(self.num_sockets):
            self.sockets[i] = set(
                map(lambda y: int(y['cpu']),
                    filter(lambda x: int(x['socket']) == i, lscpu)
                    )
            )

        # Store the lscpu output
        self._lscpu = lscpu

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

        if (cpuid < 0) or (cpuid >= self.num_cpus):
            raise TaskExit(f'a cpuid with value {cpuid} is invalid')

        if by is None:
            raise TaskExit('must specify the sibiling level')
        else:
            if by not in self._lscpu[0]:
                raise TaskExit('invalid sibiling level')

        sibiling_id = list(
            filter(lambda x: int(x['cpu']) == cpuid, self._lscpu)
        )[0][by]

        return set(
            map(
                lambda y: int(y['cpu']),
                filter(
                    lambda x: x[by] == sibiling_id,
                    self._lscpu
                )
            )
        )

    @sn.sanity_function
    def assert_consumed_cpu_set(self):
        '''Check that all the resources have been consumed.

        Tests derived from this class must implement a hook that consumes
        the cpu set as the results from the affinity tool are processed.
        '''
        return sn.assert_eq(self.cpu_set, set())

    @rfm.run_after('run')
    def parse_output(self):
        '''Extract the data from the affinity tool.'''
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
        '''Hook to control multithreading settings for each system.'''
        cp = self.current_partition.fullname
        mthread = self.system.get(cp, {}).get('multithreading', None)
        if mthread:
            self.use_multithreading = mthread

    @rfm.run_before('run')
    def set_launcher(self):
        '''Hook to control hints and cpu-bind for each system.'''
        cp = self.current_partition.fullname
        cpu_bind = self.system.get(cp, {}).get('cpu-bind', None)
        if cpu_bind:
            self.job.launcher.options += [f'--cpu-bind={cpu_bind}']

        hint = self.system.get(cp, {}).get('hint', None)
        if hint:
            self.job.launcher.options += [f'--hint={hint}']


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
    def ncpus_per_task(self):
        '''We use this property to set the hook below and keep exec order.'''
        return self.num_cpus

    @rfm.run_before('run')
    def set_num_cpus_per_task(self):
        self.num_cpus_per_task = self.ncpus_per_task

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
class OneThreadPerLogicalCoreOpenMP(AffinityOpenMPBase):
    '''Pin each OMP thread to a different logical core.'''
    parameter('omp_bind', ['threads'])

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per CPU.'

        # System-dependent settings here
        self.system = {}

    @property
    def num_omp_threads(self):
        # One OMP thread per logical core
        return self.num_cpus

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to cpus.'''
        for affinity_set in self.aff_cpus:
            # Affinity set must be of length 1 and CPU ID cannot be repeated
            if ((len(affinity_set) > 1) or not all(x in self.cpu_set
                                                   for x in affinity_set)):
                # This will force the sanity function to fail.
                raise SanityError('incorrect affinity set')

            # Decrement the affinity set from the cpu set
            self.cpu_set -= set(affinity_set)


@rfm.simple_test
class OneThreadPerPhysicalCoreOpenMP(AffinityOpenMPBase):
    '''Pin each OMP thread to a different physical core.'''
    parameter('omp_bind', ['cores'])

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per core.'

        # System-dependent settings here
        self.system = {}

    @property
    def num_omp_threads(self):
        # One OMP thread per core
        return int(self.num_cpus/self.num_cpus_per_core)

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to cores.'''
        for affinity_set in self.aff_cpus:

            # Get CPU sibilings by core
            cpu_sibilings = self.get_sibiling_cpus(affinity_set[0], by='core')

            # All CPUs in the set must belong to the same core
            if (not all(x in self.cpu_set for x in affinity_set) or
                not all(x in cpu_sibilings for x in affinity_set)):
                raise SanityError('incorrect affinity set')

            # Decrement the cpu set with all the CPUs that belong to this core
            self.cpu_set -= cpu_sibilings


@rfm.simple_test
class OneThreadPerPhysicalCoreOpenMPnomt(OneThreadPerPhysicalCoreOpenMP):
    '''Only one cpu per core booked without multithread.'''

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per core without multithreading.'
        self.use_multithreading = False

        # System-dependent settings here
        self.system = {}

    @property
    def ncpus_per_task(self):
        return self.num_omp_threads

    @rfm.run_before('sanity')
    def assert_aff_set_length(self):
        '''Only 1 CPU pinned per thread.'''
        if not all(len(aff_set) == 1 for aff_set in self.aff_cpus):
            raise SanityError('incorrect affinity set')


@rfm.simple_test
class OneThreadPerSocketOpenMP(AffinityOpenMPBase):
    '''Pin each OMP thread to a different socket.'''
    parameter('omp_bind', ['sockets'])

    def __init__(self):
        super().__init__()
        self.descr = 'Pin one OMP thread per socket.'

        # System-dependent settings here
        self.system = {}

    @property
    def num_omp_threads(self):
        # One OMP thread per core
        return self.num_sockets

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to sockets.'''
        for affinity_set in self.aff_cpus:

            # Get CPU sibilings by socket
            cpu_sibilings = self.get_sibiling_cpus(
                affinity_set[0], by='socket')

            # Alll CPUs in the affinity set must belong to the same socket
            if (not all(x in self.cpu_set for x in affinity_set) or
                not all(x in cpu_sibilings for x in affinity_set)):
                raise SanityError('incorrect affinity set')

            # Decrement all the CPUs in this socket from the cpu set.
            self.cpu_set -= cpu_sibilings


@rfm.simple_test
class OneTaskPerSocketOpenMPnomt(AffinityOpenMPBase):
    '''One task per socket, and 1 OMP thread per physical core.'''
    parameter('omp_bind', ['sockets'])
    parameter('omp_proc_bind', ['close'])

    def __init__(self):
        super().__init__()
        self.descr = 'One task per socket - wo. multithreading.'
        self.use_multithreading = False

        # System-dependent settings here
        self.system = {}

    @property
    def num_omp_threads(self):
        return int(self.num_cpus/self.num_cpus_per_core/self.num_sockets)

    @property
    def ncpus_per_task(self):
        return self.num_omp_threads

    @rfm.run_before('run')
    def set_num_tasks(self):
        self.num_tasks = self.num_sockets

    @rfm.run_before('sanity')
    def consume_cpu_set(self):

        threads_in_socket = [0]*self.num_sockets

        def get_socket_id(cpuid):
            for i in range(self.num_sockets):
                if cpuid in self.sockets[i]:
                    return i

        for affinity_set in self.aff_cpus:
            # Count the number of OMP threads that live on each socket
            threads_in_socket[get_socket_id(affinity_set[0])] += 1

            # Get CPU sibilings by socket
            cpu_sibilings = self.get_sibiling_cpus(
                affinity_set[0], by='socket'
            )

            # The size of the affinity set matches the number of OMP threads
            # and all CPUs from the set belong to the same socket.
            if ((self.num_omp_threads != len(affinity_set)) or
                not all(x in cpu_sibilings for x in affinity_set)):
                raise SanityError('incorrect affinity set')

        # Remove the sockets the cpu set.
        for i, socket in enumerate(self.sockets.values()):
            if threads_in_socket[i] == self.num_omp_threads:
                self.cpu_set -= socket


@rfm.simple_test
class OneTaskPerSocketOpenMP(OneTaskPerSocketOpenMPnomultithread):
    '''One task per socket, and as many OMP threads as CPUs per socket.

    We can reuse the test above. Just need to change the multithreading flag
    and the number of OMP threads.
    '''

    def __init__(self):
        super().__init__()
        self.descr = 'One task per socket - w. multithreading.'
        self.use_multithreading = True

        # System-dependent settings here
        self.system = {}

    @property
    def num_omp_threads(self):
        return int(self.num_cpus/self.num_sockets)


@rfm.simple_test
class ConsecutiveSocketFilling(AffinityTestBase):
    '''Fill the sockets with the tasks in consecutive order.

    This test uses as many tasks as physical cores available in a node.
    Multithreading is disabled.
    '''

    def __init__(self):
        super().__init__()
        self.use_multithreading = False

        # System-dependent settings
        self.system = {}

    @rfm.run_before('run')
    def set_tasks(self):
        self.num_tasks = int(self.num_cpus/self.num_cpus_per_core)
        self.num_cpus_per_task = 1
        self.job.launcher.options += [f'--cpu-bind=rank']

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Check that all physical cores have been used in the right order.'''
        task_count = 0
        for socket_number in range(self.num_sockets):
            # Keep track of the CPUs present in this socket
            cpus_present = set()
            for task_number in range(int(self.num_tasks/self.num_sockets)):
                # Get the list of CPUs with affinity
                affinity_set = self.aff_cpus[task_count]

                # Only 1 CPU per affinity set allowed.
                if (len(affinity_set) > 1) or (any(cpu in cpus_present
                                                   for cpu in affinity_set)):
                    raise SanityError(
                        f'incorrect affinity set for task {task_count}'
                    )

                else:
                    cpus_present.update(
                        self.get_sibiling_cpus(affinity_set[0], by='core')
                    )

                task_count += 1

            # Ensure all CPUs belong to the same socket
            cpuset_by_socket = self.get_sibiling_cpus(
                next(iter(cpus_present)), by='socket'
            )
            if not all(cpu in cpuset_by_socket for cpu in cpus_present):
                raise SanityError(
                    f'socket {socket_number} not filled in order'
                )

            else:
                # Decrement the current socket from the available CPU set
                self.cpu_set -= cpus_present


@rfm.simple_test
class AlternateSocketFilling(AffinityTestBase):
    '''Sockets are filled in a round-robin fashion.

    This test uses as many tasks as physical cores available in a node.
    Multithreading is disabled.
    '''

    def __init__(self):
        super().__init__()
        self.use_multithreading = False
        self.system = {
            # System-dependent settings here
        }

    @rfm.run_before('run')
    def set_tasks(self):
        self.num_tasks = int(self.num_cpus/self.num_cpus_per_core)
        self.num_cpus_per_task = 1
        self.num_tasks_per_socket = int(self.num_tasks/self.num_sockets)

    @rfm.run_before('sanity')
    def consume_cpu_set(self):
        '''Check that consecutive tasks are round-robin pinned to sockets.'''

        # Get a set per socket to keep track of the CPUs
        sockets = [set() for s in range(self.num_sockets)]
        task_count = 0
        for task in range(self.num_tasks_per_socket):
            for s in range(self.num_sockets):
                # Get the list of CPUs with affinity
                affinity_set = self.aff_cpus[task_count]

                # Only 1 CPU per affinity set is allowed
                if (len(affinity_set) > 1) or (any(cpu in sockets[s]
                                                   for cpu in affinity_set)):
                    raise SanityError(
                        f'incorrect affinity set for task {task_count}'
                    )

                else:
                    sockets[s].update(
                        self.get_sibiling_cpus(affinity_set[0], by='core')
                    )

                task_count += 1

            # Check that all sockets have the same CPU count
            if not all(len(s) == (task+1)*2 for s in sockets):
                self.cpu_set.add(-1)

        # Decrement the sockets from the CPU set
        for s in sockets:
            self.cpu_set -= s
