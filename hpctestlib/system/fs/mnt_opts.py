# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from string import Template

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


@rfm.simple_test
class filesystem_options_check(rfm.RunOnlyRegressionTest):
    '''filesystem mount options check

    Check if the mounted filesystems have been configured appropriately
    based on their type
    '''

    #: Reference mount options
    #:
    #: :type: `Dict[str, str]`. The key should be the file system type.
    #:   and the value should be a string with mount options.
    #:   E.g., {'xfs: 'nosuid,logbsize=32k'}
    fs_ref_opts = variable(typ.Dict, loggable=True)

    #: Fail if the test finds a filesystem type that is not in the
    #:    reference dictionary
    #:
    #: :type: `Bool`.
    #: :value: ``False``
    fail_unknown_fs = variable(typ.Bool, value=False, loggable=True)

    executable = 'cat'
    executable_opts = ['/proc/mounts']
    tags = {'system', 'fs'}

    @run_before('sanity')
    def process_system_fs_opts(self):
        self.filesystem = []
        stdout = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        with open(stdout, 'r') as fp:
            for line in fp:
                _, mnt_point, type, options, _, _ = line.split()
                self.filesystem.append((mnt_point, type, options))

    @run_before('sanity')
    def print_test_variables_to_output(self):
        '''Write the reference mount point options used by the test
        at the time of execution.
        '''
        stdout = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        with open(stdout, 'a') as fp:
            fp.write('\n---- Reference mount options ----\n')
            for mnt_type, options in self.fs_ref_opts.items():
                fp.write(f'{mnt_type} {options}\n')

    def explode_opts_str(self, opts_str):
        result = {}
        for opt in opts_str.split(','):
            if opt == '':
                continue

            opt_parts = opt.split('=', maxsplit=2)
            keystr = opt_parts[0]
            valstr = opt_parts[1] if len(opt_parts) > 1 else ''
            result[keystr] = valstr

        return result

    @sanity_function
    def assert_mnt_options(self):
        msg = Template('Found filesystem type(s) - "$mnt_type" - that are not '
                       'compatible with this test')
        msg1 = Template('The mount point "$mnt_point" of type "$mnt_type" does'
                        ' not have the "$opt" option.')
        msg2 = Template('The "$variable" variable value of "$value" does not '
                        'match the reference "$ref_value" for the "$mnt_point"'
                        ' mount point ("$mnt_type" type)')

        errors = []
        unsupported_types = set()
        for mnt_point, mnt_type, options in self.filesystem:
            opts = self.explode_opts_str(options)
            if mnt_type not in self.fs_ref_opts:
                unsupported_types.add(mnt_type)
                continue

            ref_opts = self.explode_opts_str(self.fs_ref_opts[mnt_type])
            for ref_opt, ref_value in ref_opts.items():
                if ref_opt not in opts:
                    errors.append(msg1.substitute(opt=ref_opt,
                                                  mnt_point=mnt_point,
                                                  mnt_type=mnt_type))
                elif ref_value != opts[ref_opt]:
                    errors.append(msg2.substitute(value=opts[ref_opt],
                                                  ref_value=ref_value,
                                                  variable=ref_opt,
                                                  mnt_point=mnt_point,
                                                  mnt_type=mnt_type))

        if self.fail_unknown_fs:
            errors.append(msg.substitute(
                mnt_type=', '.join(unsupported_types)))

        return sn.assert_true(errors == [], msg='\n'.join(errors))
