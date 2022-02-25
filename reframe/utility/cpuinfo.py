# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import archspec.cpu
import contextlib
import glob
import os
import re

import reframe.utility.osext as osext
from reframe.core.exceptions import SpawnedProcessError


def _bits_from_str(mask_s):
    '''Return the set bits from a string representing a bit array.'''

    bits = []
    mask = int(mask_s, 0)
    pos = 0
    while mask:
        if mask & 1:
            bits.append(pos)

        pos += 1
        mask >>= 1

    return bits


def _str_from_bits(bits):
    '''Return a string representation of a bit array with ``bits`` set.'''

    ret = 0
    for b in bits:
        ret |= (1 << b)

    return hex(ret).lower()


def _sysfs_topo():
    cache_units = {
        'K': 1024,
        'M': 1024*1024,
        'G': 1024*1024*1024
    }
    cpuinfo = {
        'topology': {}
    }
    cpu_dirs = glob.glob(r'/sys/devices/system/cpu/cpu[0-9]*')
    nodes = glob.glob(r'/sys/devices/system/node/node[0-9]*')
    cores = set()
    for cpu in cpu_dirs:
        core_cpus_path = os.path.join(cpu, 'topology/core_cpus')
        thread_siblings_path = os.path.join(cpu, 'topology/thread_siblings')
        if glob.glob(core_cpus_path):
            cores_path = core_cpus_path
        elif glob.glob(thread_siblings_path):
            cores_path = thread_siblings_path
        else:
            # Information cannot be retrieved
            continue

        with contextlib.suppress(IOError):
            with open(cores_path) as fp:
                core_cpus = fp.read()
                core_cpus = re.sub(r'[\s,]', '', core_cpus)
                core_cpus = f'0x{core_cpus.lower()}'
                cores.add(core_cpus)

    sockets = set()
    for cpu in cpu_dirs:
        package_cpus_path = os.path.join(cpu, 'topology/package_cpus')
        core_siblings_path = os.path.join(cpu, 'topology/core_siblings')
        if glob.glob(package_cpus_path):
            sockets_path = package_cpus_path
        elif glob.glob(core_siblings_path):
            sockets_path = core_siblings_path
        else:
            # Information cannot be retrieved
            continue

        with contextlib.suppress(IOError):
            with open(sockets_path) as fp:
                package_cpus = fp.read()
                package_cpus = re.sub(r'[\s,]', '', package_cpus)
                package_cpus = f'0x{package_cpus.lower()}'
                sockets.add(package_cpus)

    numa_nodes = []
    for node in nodes:
        with contextlib.suppress(IOError):
            with open(os.path.join(node, 'cpumap')) as fp:
                cpumap = fp.read()
                cpumap = re.sub(r'[\s,]', '', cpumap)
                cpumap = f'0x{cpumap.lower()}'
                numa_nodes.append(cpumap)

    numa_nodes.sort()
    caches = {}
    for cpu in cpu_dirs:
        cache_dirs = glob.glob(cpu + r'/cache/index[0-9]*')
        for cache in cache_dirs:
            cache_level = 0
            cache_size = 0
            cache_linesize = 0
            cache_associativity = 0
            cache_cpuset = ''
            with contextlib.suppress(IOError):
                with open(os.path.join(cache, 'level')) as fp:
                    cache_level = int(fp.read())

            with contextlib.suppress(IOError):
                # Skip L1 instruction cache
                with open(os.path.join(cache, 'type')) as fp:
                    if cache_level == 1 and fp.read() == 'Instruction\n':
                        continue

            with contextlib.suppress(IOError):
                with open(os.path.join(cache, 'size')) as fp:
                    cache_size = fp.read()
                    m = re.match(r'(?P<val>\d+)(?P<unit>\S)', cache_size)
                    if m:
                        value = int(m.group('val'))
                        unit = cache_units.get(m.group('unit'), 1)
                        cache_size = value*unit

            with contextlib.suppress(IOError):
                with open(os.path.join(cache, 'coherency_line_size')) as fp:
                    cache_linesize = int(fp.read())

            # Don't take the associativity directly from
            # "ways_of_associativity" file because  some archs (ia64, ppc)
            # put 0 there when fully-associative, while others (x86)
            # put something like -1.
            with contextlib.suppress(IOError):
                with open(os.path.join(cache, 'number_of_sets')) as fp:
                    cache_number_of_sets = int(fp.read())

                with open(os.path.join(cache,
                                       'physical_line_partition')) as fp:
                    cache_physical_line_partition = int(fp.read())

                if (cache_linesize and
                    cache_physical_line_partition and
                    cache_number_of_sets):
                    cache_associativity = (cache_size //
                                           cache_linesize //
                                           cache_physical_line_partition //
                                           cache_number_of_sets)

            with contextlib.suppress(IOError):
                with open(os.path.join(cache, 'shared_cpu_map')) as fp:
                    cache_cpuset = fp.read()
                    cache_cpuset = re.sub(r'[\s,]', '', cache_cpuset)
                    cache_cpuset = f'0x{cache_cpuset.lower()}'

            num_cpus = len(_bits_from_str(cache_cpuset))
            caches.setdefault((cache_level, cache_size, cache_linesize,
                               cache_associativity, num_cpus), set())
            caches[(cache_level, cache_size, cache_linesize,
                    cache_associativity, num_cpus)].add(cache_cpuset)

    num_cpus = len(cpu_dirs)
    num_cores = len(cores)
    num_sockets = len(sockets)
    num_cpus_per_core = num_cpus // num_cores if num_cores else 0
    num_cpus_per_socket = num_cpus // num_sockets if num_sockets else 0

    # Fill in the cpuinfo
    cpuinfo['num_cpus'] = num_cpus
    cpuinfo['num_cpus_per_core'] = num_cpus_per_core
    cpuinfo['num_cpus_per_socket'] = num_cpus_per_socket
    cpuinfo['num_sockets'] = num_sockets
    cpuinfo['topology']['numa_nodes'] = numa_nodes
    cpuinfo['topology']['sockets'] = sorted(list(sockets))
    cpuinfo['topology']['cores'] = sorted(list(cores))
    cpuinfo['topology']['caches'] = []
    for cache_type, cpusets in caches.items():
        (cache_level, cache_size,
         cache_linesize, cache_associativity, num_cpus) = cache_type
        c = {
            'type': f'L{cache_level}',
            'size': cache_size,
            'linesize': cache_linesize,
            'associativity': cache_associativity,
            'num_cpus': num_cpus,
            'cpusets': sorted(list(cpusets))
        }
        cpuinfo['topology']['caches'].append(c)

    return cpuinfo


def _sysctl_topo():
    try:
        exec_output = osext.run_command('sysctl hw machdep.cpu',
                                        check=True)
    except (FileNotFoundError, SpawnedProcessError):
        return {}

    cpuinfo = {
        'topology': {}
    }
    match = re.search(r'hw\.ncpu: (?P<num_cpus>\d+)', exec_output.stdout)
    if match:
        num_cpus = int(match.group('num_cpus'))

    match = re.search(r'hw\.physicalcpu: (?P<num_cores>\d+)',
                      exec_output.stdout)
    if match:
        num_cores = int(match.group('num_cores'))

    match = re.search(r'hw\.packages: (?P<num_sockets>\d+)',
                      exec_output.stdout)
    if match:
        num_sockets = int(match.group('num_sockets'))
        cpuinfo['num_sockets'] = num_sockets

    match = re.search(r'hw\.cacheconfig:(?P<cacheconfig>(\s\d+)*)',
                      exec_output.stdout)
    if match:
        cacheconfig = list(map(int, match.group('cacheconfig').split()))

    match = re.search(r'hw\.cachesize:(?P<cachesize>(\s\d+)*)',
                      exec_output.stdout)
    if match:
        cachesize = list(map(int, match.group('cachesize').split()))

    match = re.search(r'hw\.cachelinesize: (?P<linesize>\d+)',
                      exec_output.stdout)
    if match:
        linesize = int(match.group('linesize'))

    # index 0 is referring to memory
    cache_associativity = [0]
    for i in range(1, len(cachesize)):
        if cachesize[i] == 0:
            break

        match = re.search(rf'machdep\.cpu\.cache\.L{i}_associativity: '
                          rf'(?P<associativity>\d+)',
                          exec_output.stdout)
        assoc = int(match.group('associativity')) if match else 0
        cache_associativity.append(assoc)

    num_cpus_per_socket = num_cpus // num_sockets
    num_cpus_per_core = num_cpus // num_cores

    # Fill in the cpuinfo
    cpuinfo['num_cpus'] = num_cpus
    cpuinfo['num_cpus_per_socket'] = num_cpus_per_socket
    cpuinfo['num_cpus_per_core'] = num_cpus_per_core
    cpuinfo['topology']['numa_nodes'] = [_str_from_bits(range(num_cpus))]
    cpuinfo['topology']['sockets'] = [
        _str_from_bits(range(start, start+num_cpus_per_socket))
        for start in range(0, num_cpus, num_cpus_per_socket)
    ]
    cpuinfo['topology']['cores'] = [
        _str_from_bits(range(start, start+num_cpus_per_core))
        for start in range(0, num_cpus, num_cpus_per_core)
    ]
    cpuinfo['topology']['caches'] = []
    for i in range(1, len(cache_associativity)):
        t = {
            'type': f'L{i}',
            'size': cachesize[i],
            'linesize': linesize,
            'associativity': cache_associativity[i],
            'num_cpus': cacheconfig[i],
            'cpusets': [
                _str_from_bits(range(start, start+cacheconfig[i]))
                for start in range(0, num_cpus, cacheconfig[i])
            ]
        }
        cpuinfo['topology']['caches'].append(t)

    return cpuinfo


def cpuinfo():
    ret = {
        'arch': archspec.cpu.host().name
    }

    # Try first to get information from the filesystem
    if os.path.isdir('/sys'):
        topology = _sysfs_topo()
    else:
        # Try with the `sysctl` command
        topology = _sysctl_topo()

    ret.update(topology)
    return ret
