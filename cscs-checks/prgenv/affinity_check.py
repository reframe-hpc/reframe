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

def add_prgenv_nvidia(self):
    cs = self.current_system.name
    if cs in {'daint', 'dom'}:
        self.valid_prog_environs += ['PrgEnv-nvidia']


@rfm.simple_test
class CompileAffinityTool(rfm.CompileOnlyRegressionTest):
    valid_systems = [
        'daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
        'eiger:mc', 'pilatus:mc',
        'ault:amdv100'
    ]
    valid_prog_environs = [
        'PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-intel', 'PrgEnv-pgi'
    ]
    build_system = 'Make'

    # The github URL can not be specifid as `self.sourcedir` as that
    # would prevent the src folder from being copied to stage which is
    # necessary since these tests need files from it.
    sourcesdir = os.path.join('src/affinity_ref')
    prebuild_cmds = ['git clone https://github.com/vkarak/affinity']
    postbuild_cmds = ['ls affinity']
    maintainers = ['RS', 'SK']
    tags = {'production', 'scs', 'maintenance', 'craype'}

    run_after('init')(bind(add_prgenv_nvidia))

    @run_before('compile')
    def set_build_opts(self):
        self.build_system.options = ['-C affinity', 'MPI=1']

    @run_before('compile')
    def prgenv_nvidia_workaround(self):
        cs = self.current_system.name
        ce = self.current_environ.name
        if ce == 'PrgEnv-nvidia' and cs == 'dom':
            self.build_system.cppflags = [
                '-D__GCC_ATOMIC_TEST_AND_SET_TRUEVAL'
            ]

    @run_before('sanity')
    def assert_exec_exists(self):
        self.sanity_patterns = sn.assert_found(r'affinity', self.stdout)


@rfm.simple_test
class CompileAffinityToolNoOmp(CompileAffinityTool):
    valid_systems = ['eiger:mc', 'pilatus:mc']

    @run_before('compile')
    def set_build_opts(self):
        self.build_system.options = ['-C affinity', 'MPI=1', 'OPENMP=0']


class AffinityTestBase(rfm.RunOnlyRegressionTest):
    '''Base class for the affinity checks.

    It reads a reference file for each valid system, which allows this base
    class to figure out the processor's topology. The content of this reference
    file is simply the output of the `lscpu -e -J` command. For more info on
    this, see the `read_proc_topo` hook.
    '''

    # Variables to control the hint and binding options on the launcher.
    multithread = variable(bool, type(None), value=None)
    cpu_bind = variable(str, type(None), value=None)
    hint = variable(str, type(None), value=None)

    # Variable to specify system specific launcher options. This will override
    # any of the above global options.
    #
    # Example:
    #
    # system = {
    #     'daint:mc': {
    #         'multithreading': False,
    #         'cpu_bind':       'none',
    #         'hint':           'nomultithread',
    #     }
    # }
    system = variable(dict, value={})

    valid_systems = [
        'daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
        'eiger:mc', 'pilatus:mc',
        'ault:amdv100'
    ]
    valid_prog_environs = [
        'PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-intel', 'PrgEnv-pgi'
    ]

    # Dict with the partition's topology - output of "lscpu -e"
    topology = variable(dict, value={
        'dom:gpu':    'topo_dom_gpu.json',
        'dom:mc':     'topo_dom_mc.json',
        'daint:gpu':  'topo_dom_gpu.json',
        'daint:mc':   'topo_dom_mc.json',
        'eiger:mc':   'topo_eiger_mc.json',
        'pilatus:mc':   'topo_eiger_mc.json',
        'ault:amdv100': 'topo_ault_amdv100.json',
    })

    # Reference topology file as required variable
    topo_file = variable(str)

    maintainers = ['RS', 'SK']
    tags = {'production', 'scs', 'maintenance', 'craype'}

    run_after('init')(bind(add_prgenv_nvidia))

    @run_after('init')
    def set_deps(self):
        self.depends_on('CompileAffinityTool')

    @require_deps
    def set_executable(self, CompileAffinityTool):
        self.executable = os.path.join(
            CompileAffinityTool().stagedir, 'affinity/affinity'
        )

    @require_deps
    def set_topo_file(self, CompileAffinityTool):
        '''Set the topo_file variable.

        If not present in the topology dict, leave it as required.
        '''
        cp = self.current_partition.fullname
        if cp in self.topology:
            self.topo_file = os.path.join(
                CompileAffinityTool().stagedir, self.topology[cp]
            )

    # FIXME: Update the hook below once the PR #1773 is merged.
    @run_after('compile')
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
            with open(self.topo_file, 'r') as topo:
                lscpu = json.load(topo)['cpus']

        # Build the cpu set
        self.cpu_set = {int(x['cpu']) for x in lscpu}
        self.num_cpus = len(self.cpu_set)

        # Build the numa sets
        self.num_numa_nodes = len({int(x['node']) for x in lscpu})
        self.num_cpus_per_core = int(
            self.num_cpus/len({int(x['core']) for x in lscpu})
        )
        self.numa_nodes = []
        for i in range(self.num_numa_nodes):
            self.numa_nodes.append({
                int(y['cpu']) for y in [
                    x for x in lscpu if int(x['node']) == i
                ]
            })

        # Build the socket sets
        self.num_sockets = len({int(x['socket']) for x in lscpu})
        self.sockets = []
        for i in range(self.num_sockets):
            self.sockets.append({
                int(y['cpu']) for y in [
                    x for x in lscpu if int(x['socket']) == i
                ]
            })

        # Store the lscpu output
        self._lscpu = lscpu

    def get_sibling_cpus(self, cpuid, by=None):
        '''Return a cpu set where cpuid belongs to.

        The cpu set can be extracted by matching core, numa domain or socket.
        This is controlled by the `by` argument.
        '''

        if (cpuid < 0) or (cpuid >= self.num_cpus):
            raise ReframeError(f'a cpuid with value {cpuid} is invalid')

        if by is None:
            raise ReframeError('must specify the sibling level')
        else:
            if by not in self._lscpu[0]:
                raise ReframeError('invalid sibling level')

        sibling_id = [x for x in self._lscpu if int(x['cpu']) == cpuid][0][by]
        return {
            int(y['cpu']) for y in [
                x for x in self._lscpu if x[by] == sibling_id
            ]
        }

    @sn.sanity_function
    def assert_consumed_cpu_set(self):
        '''Check that all the resources have been consumed.

        Tests derived from this class must implement a hook that consumes
        the cpu set as the results from the affinity tool are processed.
        '''
        return sn.assert_eq(self.cpu_set, set())

    @run_after('run')
    def parse_output(self):
        '''Extract the data from the affinity tool.'''

        re_aff_cpus = r'CPU affinity: \[\s+(?P<cpus>[\d+\s+]+)\]'

        def parse_cpus(x):
            return sorted([int(xi) for xi in x.split()])

        with osext.change_dir(self.stagedir):
            self.aff_cpus = sn.extractall(
                re_aff_cpus, self.stdout, 'cpus', parse_cpus
            ).evaluate()

    @run_before('run')
    def set_multithreading(self):
        '''Hook to control multithreading settings for each system.'''

        cp = self.current_partition.fullname
        mthread = (
            self.system.get(cp, {}).get('multithreading', None) or
            self.multithread
        )
        if mthread:
            self.use_multithreading = mthread

    @run_before('run')
    def set_launcher(self):
        '''Hook to control hints and cpu-bind for each system.'''

        cp = self.current_partition.fullname
        cpu_bind = (
            self.system.get(cp, {}).get('cpu-bind', None) or self.cpu_bind
        )
        if cpu_bind:
            self.job.launcher.options += [f'--cpu-bind={cpu_bind}']

        hint = self.system.get(cp, {}).get('hint', None) or self.hint
        if hint:
            self.job.launcher.options += [f'--hint={hint}']

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = self.assert_consumed_cpu_set()


class AffinityOpenMPBase(AffinityTestBase):
    '''Extend affinity base with OMP hooks.

    The tests derived from this class book the full node and place the
    threads acordingly based exclusively on the OMP_BIND env var. The
    number of total OMP_THREADS will vary depending on what are we
    binding the OMP threads to (e.g. if we bind to sockets, we'll have as
    many threads as sockets).
    '''

    omp_bind = variable(str)
    omp_proc_bind = variable(str, value='spread')
    num_tasks = 1

    @property
    def ncpus_per_task(self):
        '''We use this property to set the hook below and keep exec order.'''
        return self.num_cpus

    @run_before('run')
    def set_num_cpus_per_task(self):
        self.num_cpus_per_task = self.ncpus_per_task

    @run_before('run')
    def set_omp_vars(self):
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_omp_threads),
            'OMP_PLACES': self.omp_bind,
            'OMP_PROC_BIND': self.omp_proc_bind,
        }

    @run_before('sanity')
    def consume_cpu_set(self):
        raise NotImplementedError('this function must be overridden')


@rfm.simple_test
class OneThreadPerLogicalCoreOpenMP(AffinityOpenMPBase):
    '''Pin each OMP thread to a different logical core.'''

    omp_bind = 'threads'
    descr = 'Pin one OMP thread per CPU.'

    @property
    def num_omp_threads(self):
        # One OMP thread per logical core
        return self.num_cpus

    @run_before('sanity')
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

    omp_bind = 'cores'
    descr = 'Pin one OMP thread per core.'

    @property
    def num_omp_threads(self):
        # One OMP thread per core
        return int(self.num_cpus/self.num_cpus_per_core)

    @run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to cores.'''
        for affinity_set in self.aff_cpus:

            # Get CPU siblings by core
            cpu_siblings = self.get_sibling_cpus(affinity_set[0], by='core')

            # All CPUs in the set must belong to the same core
            if (not all(x in self.cpu_set for x in affinity_set) or
                not all(x in cpu_siblings for x in affinity_set)):
                raise SanityError('incorrect affinity set')

            # Decrement the cpu set with all the CPUs that belong to this core
            self.cpu_set -= cpu_siblings


@rfm.simple_test
class OneThreadPerPhysicalCoreOpenMPnomt(OneThreadPerPhysicalCoreOpenMP):
    '''Only one cpu per core booked without multithread.'''

    use_multithreading = False
    descr = 'Pin one OMP thread per core wo. multithreading.'

    @property
    def ncpus_per_task(self):
        return self.num_omp_threads

    @run_before('sanity')
    def assert_aff_set_length(self):
        '''Only 1 CPU pinned per thread.'''
        if not all(len(aff_set) == 1 for aff_set in self.aff_cpus):
            raise SanityError('incorrect affinity set')


@rfm.simple_test
class OneThreadPerSocketOpenMP(AffinityOpenMPBase):
    '''Pin each OMP thread to a different socket.'''

    omp_bind = 'sockets'
    descr = 'Pin one OMP thread per socket.'

    @property
    def num_omp_threads(self):
        # One OMP thread per core
        return self.num_sockets

    @run_before('sanity')
    def consume_cpu_set(self):
        '''Threads are bound to sockets.'''
        for affinity_set in self.aff_cpus:

            # Get CPU siblings by socket
            cpu_siblings = self.get_sibling_cpus(
                affinity_set[0], by='socket')

            # Alll CPUs in the affinity set must belong to the same socket
            if (not all(x in self.cpu_set for x in affinity_set) or
                not all(x in cpu_siblings for x in affinity_set)):
                raise SanityError('incorrect affinity set')

            # Decrement all the CPUs in this socket from the cpu set.
            self.cpu_set -= cpu_siblings


@rfm.simple_test
class OneTaskPerSocketOpenMPnomt(AffinityOpenMPBase):
    '''One task per socket, and 1 OMP thread per physical core.'''

    omp_bind = 'sockets'
    omp_proc_bind = 'close'
    descr = 'One task per socket - wo. multithreading.'
    use_multithreading = False

    @property
    def num_omp_threads(self):
        return int(self.num_cpus/self.num_cpus_per_core/self.num_sockets)

    @property
    def ncpus_per_task(self):
        return self.num_omp_threads

    @run_before('run')
    def set_num_tasks(self):
        self.num_tasks = self.num_sockets

    @run_before('sanity')
    def consume_cpu_set(self):

        threads_in_socket = [0]*self.num_sockets

        def get_socket_id(cpuid):
            for i in range(self.num_sockets):
                if cpuid in self.sockets[i]:
                    return i

        for affinity_set in self.aff_cpus:
            # Count the number of OMP threads that live on each socket
            threads_in_socket[get_socket_id(affinity_set[0])] += 1

            # Get CPU siblings by socket
            cpu_siblings = self.get_sibling_cpus(
                affinity_set[0], by='socket'
            )

            # The size of the affinity set matches the number of OMP threads
            # and all CPUs from the set belong to the same socket.
            if ((self.num_omp_threads != len(affinity_set)) or
                not all(x in cpu_siblings for x in affinity_set)):
                raise SanityError('incorrect affinity set')

        # Remove the sockets the cpu set.
        for i, socket in enumerate(self.sockets):
            if threads_in_socket[i] == self.num_omp_threads:
                self.cpu_set -= socket


@rfm.simple_test
class OneTaskPerSocketOpenMP(OneTaskPerSocketOpenMPnomt):
    '''One task per socket, and as many OMP threads as CPUs per socket.

    We can reuse the test above. Just need to change the multithreading flag
    and the number of OMP threads.
    '''

    descr = 'One task per socket - w. multithreading.'
    use_multithreading = True

    @property
    def num_omp_threads(self):
        return int(self.num_cpus/self.num_sockets)


@rfm.simple_test
class ConsecutiveSocketFilling(AffinityTestBase):
    '''Fill the sockets with the tasks in consecutive order.

    This test uses as many tasks as physical cores available in a node.
    Multithreading is disabled.
    '''

    cpu_bind = 'rank'
    use_multithreading = False

    @run_before('run')
    def set_tasks(self):
        self.num_tasks = int(self.num_cpus/self.num_cpus_per_core)
        self.num_cpus_per_task = 1

    @run_before('sanity')
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
                        self.get_sibling_cpus(affinity_set[0], by='core')
                    )

                task_count += 1

            # Ensure all CPUs belong to the same socket
            cpuset_by_socket = self.get_sibling_cpus(
                next(iter(cpus_present)), by='socket'
            )
            if (not all(cpu in cpuset_by_socket for cpu in cpus_present) and
                len(cpuset_by_socket) == len(cpus_present)):
                raise SanityError(
                    f'socket {socket_number} not filled in order'
                )

            else:
                # Decrement the current NUMA node from the available CPU set
                self.cpu_set -= cpus_present


@rfm.simple_test
class AlternateSocketFilling(AffinityTestBase):
    '''Sockets are filled in a round-robin fashion.

    This test uses as many tasks as physical cores available in a node.
    Multithreading is disabled.
    '''

    use_multithreading = False

    @run_before('run')
    def set_tasks(self):
        self.num_tasks = int(self.num_cpus/self.num_cpus_per_core)
        self.num_cpus_per_task = 1
        self.num_tasks_per_socket = int(self.num_tasks/self.num_sockets)

    @run_before('sanity')
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
                if ((len(affinity_set) > 1) or
                    (any(cpu in sockets[s] for cpu in affinity_set)) or
                    (any(cpu not in self.sockets[s] for cpu in affinity_set))):
                    raise SanityError(
                        f'incorrect affinity set for task {task_count}'
                    )

                else:
                    sockets[s].update(
                        self.get_sibling_cpus(affinity_set[0], by='core')
                    )

                task_count += 1

            # Check that all sockets have the same CPU count
            if not all(len(s) == (task+1)*2 for s in sockets):
                self.cpu_set.add(-1)

        # Decrement the socket set from the CPU set
        for s in sockets:
            self.cpu_set -= s


@rfm.simple_test
class OneTaskPerNumaNode(AffinityTestBase):
    '''Place a task on each NUMA node.

    The trick here is to "pad" the tasks with --cpus-per-task.
    The same could be done to target any cache level instead.
    Multithreading is disabled.
    '''

    valid_systems = ['eiger:mc', 'pilatus:mc']
    use_multithreading = False
    num_cpus_per_task = required

    @run_after('init')
    def set_deps(self):
        self.depends_on('CompileAffinityToolNoOmp')

    @require_deps
    def set_executable(self, CompileAffinityToolNoOmp):
        self.executable = os.path.join(
            CompileAffinityToolNoOmp().stagedir, 'affinity/affinity'
        )

    @require_deps
    def set_topo_file(self, CompileAffinityToolNoOmp):
        '''Set the topo_file variable.

        If not present in the topology dict, leave it as required.
        '''
        cp = self.current_partition.fullname
        if cp in self.topology:
            self.topo_file = os.path.join(
                CompileAffinityToolNoOmp().stagedir, self.topology[cp]
            )

    @run_before('run')
    def set_tasks(self):
        self.num_tasks = self.num_numa_nodes
        if self.current_partition.fullname in {'eiger:mc', 'pilatus:mc'}:
            self.num_cpus_per_task = 16

    @run_before('sanity')
    def consume_cpu_set(self):
        '''Check that each task lives in a different NUMA node.'''

        if len(self.aff_cpus) != self.num_numa_nodes:
            raise SanityError(
                'number of tasks does not match the number of numa nodes'
            )

        for numa_node, aff_set in enumerate(self.aff_cpus):
            cpuset_by_numa = self.get_sibling_cpus(aff_set[0], by='node')
            if (len(aff_set) != self.num_cpus_per_task or
                any(cpu not in cpuset_by_numa for cpu in aff_set)):
                raise SanityError(
                    f'incorrect affinity set for numa node {numa_node}'
                )
            else:
                # Decrement the current NUMA node from the available CPU set
                self.cpu_set -= cpuset_by_numa
