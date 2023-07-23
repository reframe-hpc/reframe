# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


def get_mounted_fs():
    result = []
    with open('/proc/mounts','r') as f:
        for line in f.readlines():
            mnt_fs = dict()
            _, mnt_point, type, options, _, _ = line.split()
            result.append((mnt_point, type, options))
    return result


@rfm.simple_test
class filesystem_options_check(rfm.RunOnlyRegressionTest):
    '''filesystem mount options check

    Check if the mounted filesystems have been configured appropriately
    based on their type
    '''

    #: Parameter pack with listing the mounted file systems, mount points
    #: type and mount options
    #:
    #: :type: `Tupe[str, str, str]`. The keys should be
    #:   'mount_poinnt' with the filesystem mount point as a str value
    #:   'type' with the filesystem type as a str value
    #:   'options' with the filesystem options as a List[str] value
    #:   E.g., '/home', 'xfs', 'rw,nosuid,logbsize=32k'
    #:
    #: :values: Values are generated using the `/proc/mounts` file
    filesystems = parameter(get_mounted_fs(),
                            fmt=lambda x: x[0],
                            loggable=True)

    #: Reference mount options
    #:
    #: :type: `Dict[str, str]`. The key should be the file system type.
    #:   and the value should be a string with options.
    #:   E.g., {'xfs: 'nosuid,logbsize=32k'}
    #: :default: ``{}``
    fs_ref_opts = variable(typ.Dict, value={}, loggable=True)

    executable = 'stat'
    tags = {'system', 'fs'}

    @loggable
    @property
    def fs_mnt_point(self):
        '''The file system mount point

        :type: :class:`str`
        '''
        return self.filesystems[0]

    @loggable
    @property
    def fs_type(self):
        '''The file system type

        :type: :class:`str`
        '''
        return self.filesystems[1]

    @loggable
    @property
    def fs_mnt_opts(self):
        '''The file system mount options

        :type: :class:`str`
        '''
        return self.filesystems[2]

    @run_after('init')
    def set_fs_opts(self):
        # skip the test if the filesystem is not supported by the test
        self.skip_if(self.fs_type not in self.fs_ref_opts,
                     msg=f'This test does not support filesystem '
                         f'type {self.fs_type}')

        self.ref_opts=self.explode_opts_str(self.fs_ref_opts[self.fs_type])
        self.curr_opts=self.explode_opts_str(self.fs_mnt_opts)

    @run_after('init')
    def set_executable_opts(self):
        self.executable_opts = [self.fs_mnt_point]

    def explode_opts_str(self, opts_str):
        result=dict()
        for opt in opts_str.split(','):
            if opt == '':
                continue
            opt_splitted = opt.split('=')
            keystr = opt_splitted[0]
            valstr = opt_splitted[1] if len(opt_splitted) > 1 else ''
            result[keystr] = valstr
        return result

    @sanity_function
    def assert_config(self):
        skip_me = sn.extractall('No such file or directory', self.stderr)
        self.skip_if(skip_me, msg=f'Skipping test because filesystem '
                                  f'{self.fs_mnt_point} was not found')

        return sn.all(sn.chain(
            sn.map(lambda x: sn.assert_in(x, self.curr_opts,
                                          msg=f'Option {x} not present for '
                                              f'mount point '
                                              f'{self.fs_mnt_point}, '
                                              f'of type {self.fs_type}'),
                   self.ref_opts),
            sn.map(lambda x: sn.assert_eq(x[1], self.curr_opts[x[0]],
                                          msg=f'Value {x[1]} not equal to '
                                              f'{self.curr_opts[x[0]]} in '
                                              f'the variable {x[0]} for the '
                                              f'mount point '
                                              f'{self.fs_mnt_point}, '
                                              f'of type {self.fs_type}'),
                   self.ref_opts.items()),
        ))
