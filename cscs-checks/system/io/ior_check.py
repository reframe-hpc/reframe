import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class IorCheck(RegressionTest):
    def __init__(self, name, fs_mount_point, **kwargs):
        super().__init__('%s_%s' % (name, os.path.basename(fs_mount_point)),
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'IOR check (%s)' % fs_mount_point
        self.tags = {'ops', fs_mount_point}

        if fs_mount_point == '/scratch/snx1600':
            self.valid_systems = ['daint:gpu']
            self.num_tasks = 2400
            self.num_tasks_per_node = 12
        elif fs_mount_point == '/scratch/snx1600tds':
            self.valid_systems = ['dom:gpu']
            self.num_tasks = 192
            self.num_tasks_per_node = 12
        elif fs_mount_point == '/scratch/snx2000':
            self.valid_systems = ['dom:gpu']
            if self.current_system.name == 'dom':
                self.num_tasks = 192
                self.num_tasks_per_node = 12
            else:
                self.num_tasks = 1
                self.num_tasks_per_node = 1
        elif fs_mount_point == '/scratch/snx2000tds':
            self.valid_systems = ['dom:gpu']
            if self.current_system.name == 'dom':
                self.num_tasks = 192
                self.num_tasks_per_node = 12
            else:
                self.num_tasks = 1
                self.num_tasks_per_node = 1
        elif fs_mount_point == '/scratch/snx3000':
            self.valid_systems = ['daint:gpu']
            self.num_tasks = 720
            self.num_tasks_per_node = 12
        elif fs_mount_point == '/users':
            self.valid_systems = ['daint:gpu', 'dom:gpu', 'monch:compute']
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.tags |= {'maintenance'}
        elif fs_mount_point == '/apps':
            self.valid_systems = ['daint:gpu', 'dom:gpu', 'monch:compute']
            self.num_tasks = 1
            self.num_tasks_per_node = 1
        elif fs_mount_point == '/mnt/lnec':
            self.valid_systems = ['monch:compute']

        self.valid_prog_environs = ['PrgEnv-cray']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'IOR')
        self.executable = os.path.join('src', 'C', 'IOR')
        self.num_gpus_per_node = 0
        self.fs_mount_point = fs_mount_point
        self.maintainers = ['SO', 'MP']
        self.fs_reference = {
            '/scratch/snx1600': {
                'read_bw': (64326, -0.2, None),
                'write_bw': (151368, -0.2, None)
            },
            '/scratch/snx1600tds': {
                'read_bw': (1, -0.5, None),
                'write_bw': (1, -0.5, None)
            },
            '/scratch/snx2000': {
                'read_bw': (12310, -0.2, None),
                'write_bw': (11380, -0.2, None)
            },
            '/scratch/snx2000tds': {
                'read_bw': (1, -0.5, None),
                'write_bw': (1, -0.5, None)
            },
            '/scratch/snx3000': {
                'read_bw': (70753, -0.2, None),
                'write_bw': (83552, -0.2, None)
            },
            '/apps': {
                'read_bw': (204, -1.0, None),
                'write_bw': (319, -1.0, None)
            },
            '/users': {
                'read_bw': (83, -1.0, None),
                'write_bw': (304, -1.0, None)
            },
            '/mnt/lnec': {
                'read_bw': (0, None, None),
                'write_bw': (0, None, None)
            }
        }

        # Default umask is 0022, which generates file permissions -rw-r--r--
        # we want -rw-rw-r-- so we set umask to 0002
        os.umask(2)
        self.time_limit = (0, 7, 0)
        # Our references are based on fs types but regression needs reference
        # per system.
        self.reference = {
            '*': self.fs_reference[self.fs_mount_point]
        }

    def compile(self):
        super().compile(options='posix mpiio')


class IorReadCheck(IorCheck):
    def __init__(self, fs_mount_point, ior_type, **kwargs):
        super().__init__('ior_read_check', fs_mount_point, **kwargs)

        self.test_file = os.path.join(self.fs_mount_point, '.ior', 'read',
                                      'ior_write.dat')
        if ior_type == 'MPIIO':
            self.executable_opts = ('-r -a MPIIO -B -E -F -t 64m -b 32g '
                                    '-D 300 -k -o %s' % self.test_file).split()
        elif ior_type == 'POSIX':
            self.executable_opts = ('-r -a POSIX -B -E -F -t 1m -b 100m -D 60 '
                                    '-k -o %s' % self.test_file).split()

        self.sanity_patterns = sn.assert_found(r'^Max Read: ', self.stdout)
        self.perf_patterns = {
            'read_bw': sn.extractsingle(
                r'^Max Read:\s+(?P<read_bw>\S+) MiB/sec', self.stdout,
                'read_bw', float)
        }
        self.tags |= {'read'}


class IorWriteCheck(IorCheck):
    def __init__(self, fs_mount_point, ior_type, **kwargs):
        super().__init__('ior_write_check', fs_mount_point, **kwargs)
        self.test_file = os.path.join(self.fs_mount_point, '.ior', 'write',
                                      'ior_write.dat')
        if ior_type == 'MPIIO':
            self.executable_opts = ('-w -a MPIIO -B -E -F -t 64m -b 46g '
                                    '-D 300  -o %s' % self.test_file).split()
        elif ior_type == 'POSIX':
            self.executable_opts = ('-w -a POSIX -B -E -F -t 1m -b 100m -D 60 '
                                    ' -o %s' % self.test_file).split()

        self.sanity_patterns = sn.assert_found(r'^Max Write: ', self.stdout)
        self.perf_patterns = {
            'write_bw': sn.extractsingle(
                r'^Max Write:\s+(?P<write_bw>\S+) MiB/sec', self.stdout,
                'write_bw', float)
        }
        self.tags |= {'write'}


class IoMonchAcceptanceBase(IorCheck):
    def __init__(self, name, fs_mount_point, ior_type, num_tasks, **kwargs):
        super().__init__(name, fs_mount_point, **kwargs)
        self.test_file = os.path.join(self.fs_mount_point, os.getenv('USER'),
                                      'ior_write.dat')
        self.valid_systems = ['monch:compute']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = num_tasks
        self.num_tasks_per_node = 20
        reference_by_num_tasks = {
            40: {
                'read_bw': (4343.66, -0.2, None),
                'write_bw': (3625.23, -0.2, None)
            },
            80: {
                'read_bw': (9208.83, -0.2, None),
                'write_bw': (6725.39, -0.2, None)
            },
            160: {
                'read_bw': (12365.53, -0.2, None),
                'write_bw': (8073.84, -0.2, None)
            },
        }
        self.reference = {
            'monch:compute': reference_by_num_tasks[self.num_tasks]
        }
        self.tags = {'monch_acceptance'}


class IorReadScratchMonchAcceptanceCheck(IoMonchAcceptanceBase):
    def __init__(self, fs_mount_point, ior_type, num_tasks, **kwargs):
        super().__init__('ior_read_check_monch_%s_tasks' % num_tasks,
                         fs_mount_point, ior_type, num_tasks, **kwargs)
        if ior_type == 'MPIIO':
            self.executable_opts = ('-r -a MPIIO -B -E -F -t 16m -b 8g '
                                    '-D 10 -k -o %s' % self.test_file).split()
        self.sanity_patterns = sn.assert_found(r'^Max Read: ', self.stdout)
        self.perf_patterns = {
            'read_bw': sn.extractsingle(
                r'^Max Read:\s+(?P<read_bw>\S+) MiB/sec', self.stdout,
                'read_bw', float)
        }
        self.tags |= {'read'}


class IorWriteScratchMonchAcceptanceCheck(IoMonchAcceptanceBase):
    def __init__(self, fs_mount_point, ior_type, num_tasks, **kwargs):
        super().__init__('ior_write_check_monch_%s_tasks' % num_tasks,
                         fs_mount_point, ior_type, num_tasks, **kwargs)
        if ior_type == 'MPIIO':
            self.executable_opts = ('-w -a MPIIO -B -E -F -t 16m -b 8g '
                                    '-o %s' % self.test_file).split()
        self.sanity_patterns = sn.assert_found(r'^Max Write: ', self.stdout)
        self.perf_patterns = {
            'write_bw': sn.extractsingle(
                r'^Max Write:\s+(?P<write_bw>\S+) MiB/sec', self.stdout,
                'write_bw', float)
        }
        self.tags |= {'write'}


def _get_checks(**kwargs):
    ret = [IorReadCheck('/scratch/snx1600', 'MPIIO', **kwargs),
           IorReadCheck('/scratch/snx1600tds', 'MPIIO', **kwargs),
           IorReadCheck('/scratch/snx2000', 'MPIIO', **kwargs),
           IorReadCheck('/scratch/snx2000tds', 'MPIIO', **kwargs),
           IorReadCheck('/scratch/snx3000', 'MPIIO', **kwargs),
           IorReadCheck('/users', 'POSIX', **kwargs),
           IorReadCheck('/apps', 'POSIX', **kwargs),
           IorWriteCheck('/scratch/snx1600', 'MPIIO', **kwargs),
           IorWriteCheck('/scratch/snx1600tds', 'MPIIO', **kwargs),
           IorWriteCheck('/scratch/snx2000', 'MPIIO', **kwargs),
           IorWriteCheck('/scratch/snx2000tds', 'MPIIO', **kwargs),
           IorWriteCheck('/scratch/snx3000', 'MPIIO', **kwargs),
           IorWriteCheck('/users', 'POSIX', **kwargs),
           IorWriteCheck('/apps', 'POSIX', **kwargs)]
    for tasks in [40, 80, 160]:
        ret.append(IorWriteScratchMonchAcceptanceCheck(
            '/mnt/lnec', 'MPIIO', tasks, **kwargs))
        ret.append(IorReadScratchMonchAcceptanceCheck(
            '/mnt/lnec', 'MPIIO', tasks, **kwargs))
    return ret
