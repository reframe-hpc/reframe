# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# rfmdocstart: stream
import reframe as rfm
import reframe.utility.sanity as sn
import inspect
import reframe.core.config as cfg
import pprint

@rfm.simple_test
class StreamMultiSysTest(rfm.RegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['cray', 'gnu', 'gnu-azhpc', 'intel', 'pgi']
    prebuild_cmds = [
        'wget https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c'  # noqa: E501
    ]
    build_system = 'SingleSource'
    sourcepath = 'stream.c'
    executable_opts = [
                       '-nRep 50', \
                       '-codeAlg 7', \
                       '-cores 0 1 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 61 64 68 72 76 80 84 88 92 96 100 104 108 112 116 119', \
                       '-memMB 4000', \
                       '-alignKB 4096', \
                       '-aPadB 0', \
                       '-bPadB 0', \
                       '-cPadB 0', \
                       '-testMask 15'
    ]
    variables = {
        'OMP_NUM_THREADS': '4',
        'OMP_PLACES': 'cores'
    }

    # Flags per programming environment
    flags = variable(dict, value={
        'gnu-azhpc':   ['-fopenmp', '-O3', '-Wall']
    })


    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cppflags = ['-DSTREAM_ARRAY_SIZE=$((1 << 25))']
        self.build_system.cflags = ['-fopenmp', '-O3', '-Wall']
        environ = self.current_environ.name

    @run_before('run')
    def set_num_threads(self):
        #print("Name: {}".format(self.current_system.name))
        #print("variables: {}".format(self.variables))
        #print("=====================")
        #pprint.pprint(vars(self.current_system))
        #print("=========------------============")
        vm_info = self.current_system.node_data
        #print("vm_info: {}".format(vm_info))
        core_count = 1
        if vm_info != None and 'capabilities' in vm_info:
            vcpus = int(vm_info['capabilities']['vCPUs'])
            vcpus_per_core = int(vm_info['capabilities']['vCPUsPerCore'])
            cores = int(vcpus/vcpus_per_core)
        num_threads = cores
        self.num_cpus_per_task = num_threads
        self.variables = {
            'OMP_NUM_THREADS': str(num_threads),
            'OMP_PLACES': 'cores'
        }

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def extract_bw(self, kind='Copy'):
        if kind not in {'Copy', 'Scale', 'Add', 'Triad'}:
            raise ValueError(f'illegal value in argument kind ({kind!r})')

        return sn.extractsingle(rf'{kind}:\s+(\S+)\s+.*',
                                self.stdout, 1, float)

    @run_before('performance')
    def set_perf_variables(self):
        #print("Name: {}".format(self.current_system.name))
        #print("variables: {}".format(self.variables))
        #print("=====================")
        #pprint.pprint(vars(self.current_system))
        #print("=========------------============")
        
        vm_info = self.current_system.node_data
        #print("vm_info: {}".format(vm_info))
        if vm_info != None and 'nhc_values' in vm_info:
            self.reference = {
                vm_info['vm_series']: {
                    'Triad': (
                        vm_info['nhc_values']['stream_triad'],
                        vm_info['nhc_values']['stream_triad_limits'][0],
                        vm_info['nhc_values']['stream_triad_limits'][1],
                        'MB/s'
                    )
                }
            }
            #print("reference: {}".format(self.reference))
            #print("current partition name: {}".format(self.current_partition.fullname))
            #print("build system: {}".format(self.build_system))
            self.perf_variables = {
                'Copy': self.extract_bw(),
                'Scale': self.extract_bw('Scale'),
                'Add': self.extract_bw('Add'),
                'Triad': self.extract_bw('Triad'),
            }
        else:
            print("vm_info == None or nhc_values is not a key")
# rfmdocend: stream
