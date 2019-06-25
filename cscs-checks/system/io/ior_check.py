import os
import getpass

import reframe as rfm
import reframe.utility.sanity as sn


class IorCheck(rfm.RegressionTest):
    def __init__(self, base_dir):
        super().__init__()
        self.descr = 'IOR check (%s)' % base_dir
        self.tags = {'ops', base_dir}

        self.base_dir = base_dir
        self.username = getpass.getuser()
        self.test_dir = os.path.join(self.base_dir, self.username, '.ior')
        self.test_file = os.path.join(self.test_dir, 'ior.dat')

        try:
            os.mkdir(self.test_dir)
        except OSError:
            pass

        self.fs = {
            '/scratch/snx1600tds': {
                'valid_systems': ['dom:gpu'],
                'dom': {
                    'num_tasks': 2,
                    # We will be using 1 task per node to avoid cache 
                    # effects on read. The option "-C" could be used 
                    # with many tasks per node. 8 tasks are enough
                    # to get ~peak perf (write 5.4 GB/s, read 4.3 GB/s)
                    }
                },
            '/scratch/snx1600': {
                'valid_systems': ['daint:gpu'],
                'daint': {
                    'num_tasks': 1,  # to validate
                    }
                },
            '/scratch/snx3000tds': {
                'valid_systems': ['dom:gpu'],
                'dom': {
                    'num_tasks': 2,  # to validate
                    }
                },
            '/scratch/snx3000': {
                'valid_systems': ['daint:gpu'],
                'daint': {
                    'num_tasks': 1,  # to validate
                    }
                },
            '/users': {
                'valid_systems': ['daint:gpu', 'dom:gpu', 'fulen:normal'],
                'ior_block_size': '8g',
                'daint': {
                    'num_tasks': 1,  # to validate
                    },
                'dom': {
                    'num_tasks': 1,  # to validate
                    },
                'fulen': {
                    'num_tasks': 1,
                    'build_system_cc': 'mpicc',
                    'build_system_cxx': 'mpic++',
                    'valid_prog_environs': ['PrgEnv-gnu']
                    }
                },
            '/scratch/shared/fulen': {
                'valid_systems': ['fulen:normal'],
                'ior_block_size': '48g',
                'fulen': {
                    'num_tasks': 8,
                    'build_system_cc': 'mpicc',
                    'build_system_cxx': 'mpic++',
                    'valid_prog_environs': ['PrgEnv-gnu']
                    }
                }
            }

        # Setting some default values
        for data in self.fs.values():
            data.setdefault('ior_block_size', '24g')
            data.setdefault('ior_access_type', 'MPIIO')
            data.setdefault('reference', 
                            {'read_bw': (1, -0.5, None),
                             'write_bw': (1, -0.5, None)
                             })
            
        self.valid_systems = self.fs[base_dir]['valid_systems']

        cur_sys = self.current_system.name

        try:
            self.num_tasks = self.fs[base_dir][cur_sys]['num_tasks']
        except KeyError:
            self.num_tasks = 1
        try:
            tasks_per_node = self.fs[base_dir][cur_sys]['num_tasks_per_node']
            self.num_tasks_per_node = tasks_per_node
        except KeyError:
            self.num_tasks_per_node = 1

        self.ior_block_size = self.fs[base_dir]['ior_block_size']
        self.ior_access_type = self.fs[base_dir]['ior_access_type']
        self.executable_opts = ['-B', '-F', '-C ', '-Q 1', '-t 4m', '-D 30',
                                '-b', self.ior_block_size,
                                '-a', self.ior_access_type,
                                '-o', self.test_file]
        self.sourcesdir = os.path.join(self.current_system.resourcesdir, 'IOR')
        self.executable = os.path.join('src', 'C', 'IOR')
        self.build_system = 'Make'

        try:
            prog_env = self.fs[base_dir][cur_sys]['valid_prog_environs']
            self.valid_prog_environs = prog_env
        except KeyError:
            self.valid_prog_environs = ['PrgEnv-cray']

        try:
            self.build_system.cc = self.fs[base_dir][cur_sys]['build_system_cc']
        except KeyError:
            pass
        try:
            self.build_system.cxx = self.fs[base_dir][cur_sys]['build_system_cxx']
        except KeyError:
            pass

        self.build_system.options = ['posix', 'mpiio']
        self.build_system.max_concurrency = 1
        self.num_gpus_per_node = 0

        # Default umask is 0022, which generates file permissions -rw-r--r--
        # we want -rw-rw-r-- so we set umask to 0002
        os.umask(2)
        self.time_limit = (0, 5, 0)
        # Our references are based on fs types but regression needs reference
        # per system.
        self.reference = {
            '*': self.fs[base_dir]['reference']
        }

        self.maintainers = ['SO', 'GLR']

        if self.current_system.name == 'dom':
            self.tags = {'production'}


@rfm.parameterized_test(
    ['/scratch/snx1600tds'],
    ['/scratch/snx1600'],
    ['/scratch/snx3000tds'],
    ['/scratch/snx3000'],
    ['/users'],
    ['/scratch/shared/fulen']
)
class IorWriteCheck(IorCheck):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.executable_opts.extend(['-w', '-k'])
        self.sanity_patterns = sn.assert_found(r'^Max Write: ', self.stdout)
        self.perf_patterns = {
            'write_bw': sn.extractsingle(
                r'^Max Write:\s+(?P<write_bw>\S+) MiB/sec', self.stdout,
                'write_bw', float)
        }
        self.tags |= {'write'}


@rfm.parameterized_test(['/scratch/snx1600tds'],
                        ['/scratch/snx1600'],
                        ['/scratch/snx3000tds'],
                        ['/scratch/snx3000'],
                        ['/users'],
                        ['/scratch/shared/fulen']
                        )
class IorReadCheck(IorCheck):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.executable_opts.extend(['-r'])
        self.sanity_patterns = sn.assert_found(r'^Max Read: ', self.stdout)
        self.perf_patterns = {
            'read_bw': sn.extractsingle(
                r'^Max Read:\s+(?P<read_bw>\S+) MiB/sec', self.stdout,
                'read_bw', float)
        }
        self.tags |= {'read'}
