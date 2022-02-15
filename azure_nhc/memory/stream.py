# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# rfmdocstart: streamtest4
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
    reference = {
        'catalina': {
            'Copy':  (37800, -0.05, 0.05, 'MB/s'),
            'Scale': (35000, -0.05, 0.05, 'MB/s'),
            'Add':   (37000, -0.05, 0.05, 'MB/s'),
            'Triad': (18800, -0.05, 0.05, 'MB/s')
        }
    }

    # Flags per programming environment
    flags = variable(dict, value={
        'cray':  ['-fopenmp', '-O3', '-Wall'],
        'gnu-azhpc':   ['-fopenmp', '-O3', '-Wall'],
        'intel': ['-qopenmp', '-O3', '-Wall'],
        'pgi':   ['-mp', '-O3']
    })

    # Number of cores for each system
    cores = variable(dict, value={
        'hbrs_v3:default': 120,
        'daint:gpu': 12,
        'daint:mc': 36,
        'daint:login': 10
    })

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cppflags = ['-DSTREAM_ARRAY_SIZE=$((1 << 25))']
        self.build_system.cflags = ['-fopenmp', '-O3', '-Wall']
        environ = self.current_environ.name
        print("Name: {}".format(self.current_system.name))
        print("variables: {}".format(self.variables))
#        print("self.current_system: {}".format(self.current_system.json()))
#        print("self.current_environment: {}".format(self.current_environ))
#        print("self: {}".format(self))

#        site_config = ''
#        filename = ''
#        vm_info = {}
#        temp_test = cfg._SiteConfig(site_config, filename)
#        vm_info =  temp_test._rep_azure_vm_info()
#        print("vm_info: {}".format(vm_info))
        print("=====================")
        pprint.pprint(vars(self.current_system))
        print("=========------------============")
        vm_info = self.current_system.vm_data
        #vm_info = cfg._SiteConfig.get_vm_info()
        print("vm_info: {}".format(vm_info))
        #print("vm_series: {}".format(vm_info['vm_series']))
        self.reference = {
            vm_info['vm_series']: {
                'Copy':  (850000, None, None, 'MB/s'),
                'Scale': (870000, None, None, 'MB/s'),
                'Add':   (880000, None, None, 'MB/s'),
                'Triad': (vm_info['nhc_values']['stream_triad'], -0.05, 0.05, 'MB/s')
            }
        }
#       self.reference[vm_info['vm_series']] = {} # {'Triad': (16800, -0.05, 0.05, 'MB/s')}
        print("reference: {}".format(self.reference))
        print("current partition name: {}".format(self.current_partition.fullname))
        print("build system: {}".format(self.build_system))
#        print(rfm.core.systems.System.name)
#        method_list = [method for method in dir(cfg._SiteConfig) if method.startswith('__') is False]
#        method_list = [method for method in dir(self.build_system) if method.startswith('__') is False]
#        print("this: {}".format(method_list))

    @run_before('run')
    def set_num_threads(self):
        num_threads = self.cores.get(self.current_partition.fullname, 1)
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
        self.perf_variables = {
            'Copy': self.extract_bw(),
            'Scale': self.extract_bw('Scale'),
            'Add': self.extract_bw('Add'),
            'Triad': self.extract_bw('Triad'),
        }
# rfmdocend: streamtest4
