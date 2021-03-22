# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Managing system information.

.. versionadded:: 3.6.0

'''
import archspec.cpu
import glob
import os
import re

import reframe.utility.osext as osext
from reframe.core.exceptions import SpawnedProcessError


def bits_from_string(mask):
    ret = []
    mask_int = int(mask, 0)
    index = 0
    while mask_int:
        if mask_int & 1:
            ret.append(index)

        index += 1
        mask_int >>= 1

    return ret


def string_from_bits(ids):
    ret = 0
    for id in ids:
        ret |= (1 << id)

    return hex(ret).upper()


def filesystem_info():
    cache_units = {
        'K': 1024,
        'M': 1048576,
        'G': 1073741824
    }
    processor_info = {
        'topology': {}
    }
    cpu_dirs = glob.glob(r'/sys/devices/system/cpu/cpu[0-9]*')
    nodes = glob.glob(r'/sys/devices/system/node/node[0-9]*')

    cores = set()
    for cpu in cpu_dirs:
        with open(os.path.join(cpu, 'topology/core_cpus')) as fp:
            core_cpus = fp.read()
            core_cpus = re.sub(r'[\s,]', '', core_cpus)
            core_cpus = f'0x{core_cpus.upper()}'
            cores.add(core_cpus)

    sockets = set()
    for cpu in cpu_dirs:
        with open(os.path.join(cpu, 'topology/package_cpus')) as fp:
            package_cpus = fp.read()
            package_cpus = re.sub(r'[\s,]', '', package_cpus)
            package_cpus = f'0x{package_cpus.upper()}'
            sockets.add(package_cpus)

    numa_nodes = []
    for node in nodes:
        with open(os.path.join(node, 'cpumap')) as fp:
            cpumap = fp.read()
            cpumap = re.sub(r'[\s,]', '', cpumap)
            cpumap = f'0x{cpumap.upper()}'
            numa_nodes.append(cpumap)

    numa_nodes.sort()

    caches = {}
    for cpu in cpu_dirs:
        cache_dirs = glob.glob(cpu + r'/cache/index[0-9]*')
        for cache in cache_dirs:
            with open(os.path.join(cache, 'level')) as fp:
                cache_level = int(fp.read())

            # Skip L1 instruction cache
            with open(os.path.join(cache, 'type')) as fp:
                if cache_level == 1 and fp.read() == 'Instruction\n':
                    continue

            with open(os.path.join(cache, 'ways_of_associativity')) as fp:
                cache_associativity = int(fp.read())

            with open(os.path.join(cache, 'size')) as fp:
                cache_size = fp.read()
                m = re.match(r'(?P<val>\d+)(?P<unit>\S)', cache_size)
                if m:
                    value = int(m.group('val'))
                    unit = cache_units.get(m.group('unit'), 1)
                    cache_size = value*unit

            with open(os.path.join(cache, 'coherency_line_size')) as fp:
                cache_linesize = int(fp.read())

            with open(os.path.join(cache, 'shared_cpu_map')) as fp:
                cache_cpuset = fp.read()
                cache_cpuset = re.sub(r'[\s,]', '', cache_cpuset)
                cache_cpuset = f'0x{cache_cpuset.upper()}'

            num_cpus = len(bits_from_string(cache_cpuset))
            caches.setdefault((cache_level, cache_size, cache_linesize,
                               cache_associativity, num_cpus), set())
            caches[(cache_level, cache_size, cache_linesize,
                    cache_associativity, num_cpus)].add(cache_cpuset)

    num_cpus = len(cpu_dirs)
    num_cores = len(cores)
    num_sockets = len(sockets)
    num_cpus_per_core = num_cpus // num_cores
    num_cpus_per_socket = num_cpus // num_sockets

    processor_info['num_cpus'] = num_cpus
    processor_info['num_cpus_per_core'] = num_cpus_per_core
    processor_info['num_cpus_per_socket'] = num_cpus_per_socket
    processor_info['num_sockets'] = num_sockets
    processor_info['topology']['numa_nodes'] = numa_nodes
    processor_info['topology']['sockets'] = sorted(list(sockets))
    processor_info['topology']['cores'] = sorted(list(cores))
    processor_info['topology']['caches'] = []
    for cache_type, cpusets in caches.items():
        (cache_level, cache_size, cache_linesize, cache_associativity,
         num_cpus) = cache_type
        c = {
            'type': f'L{cache_level}',
            'size': cache_size,
            'linesize': cache_linesize,
            'associativity': cache_associativity,
            'num_cpus': num_cpus,
            'cpusets': sorted(list(cpusets))
        }
        processor_info['topology']['caches'].append(c)

    return processor_info


def sysctl_info():
    try:
        exec_output = osext.run_command('sysctl hw machdep.cpu.cache',
                                        check=True)
    except (FileNotFoundError, SpawnedProcessError):
        return {}

    processor_info = {
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
        processor_info['num_sockets'] = num_sockets

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

    cache_associativity = [0]
    # index 0 is referring to memory
    for i in range(1, len(cachesize)):
        if cachesize[i] == 0:
            break

        match = re.search(rf'machdep\.cpu\.cache\.L{i}_associativity: '
                          rf'(?P<associativity>\d+)',
                    exec_output.stdout)
        ca = int(match.group('associativity')) if match else 0
        cache_associativity.append(ca)

    num_cpus_per_socket = num_cpus // num_sockets
    num_cpus_per_core = num_cpus // num_cores

    processor_info['num_cpus'] = num_cpus
    processor_info['num_cpus_per_socket'] = num_cpus_per_socket
    processor_info['num_cpus_per_core'] = num_cpus_per_core
    processor_info['topology']['numa_nodes'] = string_from_bits(range(num_cpus))
    processor_info['topology']['sockets'] = [
        string_from_bits(range(start, start+num_cpus_per_socket)) for start
        in range(0, num_cpus, num_cpus_per_socket)
    ]
    processor_info['topology']['cores'] = [
        string_from_bits(range(start, start+num_cpus_per_core)) for start
        in range(0, num_cpus, num_cpus_per_core)
    ]
    processor_info['topology']['caches'] = []
    for i in range(1, len(cache_associativity)):
        t = {
            'type': f'L{i}',
            'size': cachesize[i],
            'linesize': linesize,
            'associativity': cache_associativity[i],
            'num_cpus': cacheconfig[i],
            'cpusets': [
                string_from_bits(range(start, start+cacheconfig[i]))
                for start in range(0, num_cpus, cacheconfig[i])
            ]
        }
        processor_info['topology']['caches'].append(t)

    return processor_info


def get_proc_info():
    processor_info = {
        'arch': archspec.cpu.host().name
    }
    # Try first to get information from the filesystem
    if glob.glob('/sys/'):
        topology_information = filesystem_info()

    # Try the `sysctl` command
    topology_information = sysctl_info()
    processor_info.update(topology_information)
    return processor_info
