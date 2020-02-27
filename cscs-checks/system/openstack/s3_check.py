# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import getpass


class S3apiCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        endpoint = 'object.cscs.ch'
        self.descr = 'S3API check for (%s)' % endpoint
        self.tags = {'ops', 'object_store'}
        self.valid_systems = ['dom:gpu', 'daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.time_limit = (0, 5, 0)
        self.maintainers = ['VH', 'GLR']
        self.executable = 's3_test.sh'
        self.username = getpass.getuser()
        if self.current_system.name in ['dom']:
            self.tags |= {'production'}


@rfm.simple_test
class S3apiCreateBucket(S3apiCheck):
    def __init__(self):
        super().__init__()
        self.executable_opts = ['s3_create_bucket.py',
                                self.current_system.name,
                                self.username]

        self.sanity_patterns = sn.assert_found(r'Average bucket creation time',
                                               self.stdout)
        self.perf_patterns = {
            'bucket_ctime': sn.extractsingle(
                r'Average bucket creation time \(s\):\s(?P<secs>\S+)',
                self.stdout, 'secs', float)
        }

        self.reference = {
            '*': {
                # below 1.1 secs is ok
                'bucket_ctime': (1, None, 0.01, 's'),
            }
        }


@rfm.simple_test
class S3apiCreateSmallObject(S3apiCheck):
    def __init__(self):
        super().__init__()
        self.executable_opts = ['s3_create_small_object.py',
                                self.current_system.name,
                                self.username]

        self.sanity_patterns = sn.assert_found(r'Average object creation time',
                                               self.stdout)
        self.perf_patterns = {
            'object_ctime': sn.extractsingle(
                r'Average object creation time \(s\):\s(?P<secs>\S+)',
                self.stdout, 'secs', float)
        }

        self.reference = {
            '*': {
                # below 1.1 secs is ok
                'object_ctime': (1, None, 0.01, 's'),
            }
        }


@rfm.simple_test
class S3apiUploadLargeObject(S3apiCheck):
    def __init__(self):
        super().__init__()
        self.executable_opts = ['s3_upload_large_object.py',
                                self.current_system.name,
                                self.username]

        self.sanity_patterns = sn.assert_found(r'Average upload rate',
                                               self.stdout)
        self.perf_patterns = {
            'upload_rate': sn.extractsingle(
                r'Average upload rate \(MiB/s\):\s(?P<rate>\S+)',
                self.stdout, 'rate', float)
        }

        self.reference = {
            '*': {
                # above 10MiB/s is ok
                'upload_rate': (10, -0.5, None, 'MiB/s'),
            }
        }


@rfm.simple_test
class S3apiDownloadLargeObject(S3apiCheck):
    def __init__(self):
        super().__init__()
        self.executable_opts = ['s3_download_large_object.py',
                                self.current_system.name,
                                self.username]

        self.sanity_patterns = sn.assert_found(r'Average download rate',
                                               self.stdout)
        self.perf_patterns = {
            'download_rate': sn.extractsingle(
                r'Average download rate \(MiB/s\):\s(?P<rate>\S+)',
                self.stdout, 'rate', float)
        }

        self.reference = {
            '*': {
                # above 10MiB/s is ok
                'download_rate': (10, -0.5, None, 'MiB/s'),
            }
        }


@rfm.simple_test
class S3apiDeleteBucketObject(S3apiCheck):
    def __init__(self):
        super().__init__()
        self.executable_opts = ['s3_delete.py',
                                self.current_system.name,
                                self.username]

        self.sanity_patterns = sn.assert_found(r'Average deletion time',
                                               self.stdout)
        self.perf_patterns = {
            'object_dtime': sn.extractsingle(
                r'Average deletion time \(s\):\s(?P<secs>\S+)',
                self.stdout, 'secs', float)
        }

        self.reference = {
            '*': {
                # below 1.1 secs is ok
                'object_dtime': (1, None, 0.01, 's'),
            }
        }
