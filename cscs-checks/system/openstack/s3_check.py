# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import getpass


class S3apiCheck(rfm.RunOnlyRegressionTest):
    descr = 'S3API check for (object.cscs.ch)'
    valid_systems = ['dom:gpu', 'daint:gpu']
    valid_prog_environs = ['builtin']
    time_limit = '5m'
    executable = 's3_test.sh'
    username = getpass.getuser()
    maintainers = ['VH', 'GLR']
    tags = {'ops', 'object_store'}

    @run_after('init')
    def add_production_tag(self):
        if self.current_system.name in {'dom'}:
            self.tags |= {'production'}


@rfm.simple_test
class S3apiCreateBucket(S3apiCheck):
    reference = {
        '*': {
            # below 1.1 secs is ok
            'bucket_ctime': (1, None, 0.01, 's'),
        }
    }

    @run_after('init')
    def set_exec_opts(self):
        self.executable_opts = ['s3_create_bucket.py',
                                self.current_system.name,
                                self.username]

    @sanity_function
    def assert_bucket_creation(self):
        return sn.assert_found(r'Average bucket creation time', self.stdout)

    @performance_function('s')
    def bucket_ctime(self):
        return sn.extractsingle(
            r'Average bucket creation time \(s\):\s(?P<secs>\S+)',
            self.stdout, 'secs', float
        )


@rfm.simple_test
class S3apiCreateSmallObject(S3apiCheck):
    reference = {
        '*': {
            # below 1.1 secs is ok
            'object_ctime': (1, None, 0.01, 's'),
        }
    }

    @run_after('init')
    def set_deps_and_exec_opts(self):
        self.depends_on('S3apiCreateBucket')
        self.executable_opts = ['s3_create_small_object.py',
                                self.current_system.name,
                                self.username]

    @sanity_function
    def assert_object_creation(self):
        return sn.assert_found(r'Average object creation time', self.stdout)

    @performance_function('s')
    def object_ctime(self):
        return sn.extractsingle(
            r'Average object creation time \(s\):\s(?P<secs>\S+)',
            self.stdout, 'secs', float
        )


@rfm.simple_test
class S3apiUploadLargeObject(S3apiCheck):
    reference = {
        '*': {
            # above 10MiB/s is ok
            'upload_rate': (10, -0.5, None, 'MiB/s'),
        }
    }

    @run_after('init')
    def set_deps_and_exec_opts(self):
        self.depends_on('S3apiCreateBucket')
        self.executable_opts = ['s3_upload_large_object.py',
                                self.current_system.name,
                                self.username]

    @sanity_function
    def assert_object_upload(self):
        return sn.assert_found(r'Average upload rate', self.stdout)

    @performance_function('MiB/s')
    def upload_rate(self):
        return sn.extractsingle(
            r'Average upload rate \(MiB/s\):\s(?P<rate>\S+)',
            self.stdout, 'rate', float
        )


@rfm.simple_test
class S3apiDownloadLargeObject(S3apiCheck):
    reference = {
        '*': {
            # above 10MiB/s is ok
            'download_rate': (10, -0.5, None, 'MiB/s'),
        }
    }

    @run_after('init')
    def set_deps_and_exec_opts(self):
        self.depends_on('S3apiUploadLargeObject')
        self.executable_opts = ['s3_download_large_object.py',
                                self.current_system.name,
                                self.username]

    @sanity_function
    def assert_object_download(self):
        return sn.assert_found(r'Average download rate', self.stdout)

    @performance_function('MiB/s')
    def download_rate(self):
        return sn.extractsingle(
            r'Average download rate \(MiB/s\):\s(?P<rate>\S+)',
            self.stdout, 'rate', float
        )


@rfm.simple_test
class S3apiDeleteBucketObject(S3apiCheck):
    reference = {
        '*': {
            # below 1.1 secs is ok
            'object_dtime': (1, None, 0.01, 's'),
        }
    }

    @run_after('init')
    def set_deps_and_exec_opts(self):
        self.depends_on('S3apiCreateSmallObject')
        self.depends_on('S3apiDownloadLargeObject')
        self.executable_opts = ['s3_delete.py',
                                self.current_system.name,
                                self.username]

    @sanity_function
    def assert_bucket_deletion(self):
        return sn.assert_found(r'Average deletion time', self.stdout)

    @performance_function('s')
    def object_dtime(self):
        return sn.extractsingle(
            r'Average deletion time \(s\):\s(?P<secs>\S+)',
            self.stdout, 'secs', float
        )
