# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.utility.versioning import VersionValidator
from reframe.core.exceptions import ReframeError


@rfm.simple_test
class trivy_scan_check(rfm.RunOnlyRegressionTest):
    '''
    filesystem scanning check

    Scan the file systems using aquasecurity/triyv
    (https://github.com/aquasecurity/trivy)

    The test uses the fs option and test against three different scanners
    vuln, config, and secret.
    '''

    #: Parameter list encoding the folders to scan.
    #:
    #: :type: :class:`str`
    #: :values: example values are '/home', '/usr/', and '/etc'
    folders = parameter([], loggable=True)

    #: The release version of the trivy to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'latest'``
    trivy_release = variable(str, value='latest', loggable=True)

    #: Download the trivy binary, instead of using an available trivy binary
    #:
    #: :type: :class:`boolean`
    #: :default: ``False``
    download_trivy = variable(bool, value=False, loggable=True)

    #: The release os and arch string for the download string
    #:
    #: :type: :class:`str`
    #: :default: ``'latest'``
    trivy_os_arch = variable(str, value='Linux-64bit', loggable=True)

    #: :type: `Tupe[str, str]`. The tuple fields represent
    #:   'scanner type' the scanners type to pass to the --scanners flag
    #:   'output title' the output title of the scanner in the output
    #:
    #: :values: ('vuln', 'Vulnerability'),
    #:          ('secret', 'Secret'),
    #:          ('config', 'Misconfiguration')
    scan_type = parameter([
        ('vuln', 'Vulnerability'),
        ('secret', 'Secret'),
        ('config', 'Misconfiguration')
    ], fmt=lambda x: x[0], loggable=True)

    # we want to guarantee that the test breaks if the schemaversion changes
    trivy_json_schema_version = 2

    executable = 'trivy'
    trivy_output_file = 'trivy.out'
    tags = {'system', 'scan'}

    @loggable
    @property
    def scanner_name(self):
        '''The scanner name.

        :type: :class:`str`
        '''
        return self.__scanner_name

    @loggable
    @property
    def scanner_type(self):
        '''The scanner type.

        :type: :class:`str`
        '''
        return self.__scanner_type

    @run_after('init')
    def unpack_scan_type(self):
        self.__scanner_type, self.__scanner_name = self.scan_type

    @run_after('init')
    def check_supported_trivy_version(self):
        # This check only supports trivy versions greater or equal to 0.37.0
        if self.trivy_release == 'latest':
            return

        validator = VersionValidator(f'>=0.37.0')
        if not validator.validate(self.trivy_release):
            raise ReframeError(f'trivy version {self.trivy_release} '
                               f'is not supported')

    @run_after('init')
    def set_files_to_keep(self):
        self.keep_files = [self.trivy_output_file]

    @run_after('init')
    def download_trivy_if_required(self):
        if not self.download_trivy:
            return

        # using GitHub API in case the version is set to latest
        # it may hit the maximum number of API calss to GitHub
        if self.trivy_release == 'latest':
            self.prerun_cmds = [
                f"__tar_file=`curl -s https://api.github.com/repos/aquasecurity/trivy/releases/latest | grep browser_download_url | grep _{self.trivy_os_arch}.tar.gz\\\" | awk '{{print $2}}' | tr -d '\"'`",  # noqa: E501
                r'curl -LJO ${__tar_file}',
                r"tar xf $(basename ${__tar_file})",
                r"rm -f $(basename ${__tar_file})"
            ]
        else:
            # downloading from the release url if version is not set to latest
            tar_file = f'trivy_{self.trivy_release}_{self.trivy_os_arch}.tar.gz'
            self.prerun_cmds = [
                f'curl -LJO https://github.com/aquasecurity/trivy/releases/download/v{self.trivy_release}/{tar_file}',  # noqa: E501
                f"tar xf {tar_file}",
                f"rm -f {tar_file}"
            ]

        self.executable = f'./{self.executable}'
        self.postrun_cmds = [
            f'rm -rf {self.executable}'
        ]

    @run_after('init')
    def prepend_fs_option_cmd_opts(self):
        self.executable_opts = ['fs', f'{self.folders}'] + self.executable_opts

    @run_before('run')
    def update_executable_opts(self):
        self.executable_opts += [
            '--timeout', self.time_limit if self.time_limit else '5000m',
            '--cache-dir', self.stagedir,
            '-f', 'json',
            '-o', self.trivy_output_file
        ]

    @run_before('run')
    def set_scanner(self):
        self.executable_opts += [
            '--scanners', f'{self.__scanner_type}'
        ]

    def load_json_output(self, filename):
        with open(filename) as fp:
            try:
                trivy_output_json = json.loads(fp.read())
                return trivy_output_json
            except json.JSONDecodeError as e:
                return None

    @deferrable
    def assert_supported_schemaversion(self, trivy_output_json):
        field_name = 'SchemaVersion'
        return sn.and_(
            sn.assert_in(field_name, trivy_output_json,
                         msg=f'{field_name} not found in trivy output. You may '
                             f'have a corrupt {self.trivy_output_file} file'),
            sn.assert_eq(self.trivy_json_schema_version,
                         trivy_output_json[field_name],
                         msg=f'{field_name} value not supported. Found '
                             f'{trivy_output_json[field_name]}, expected '
                             f'{self.trivy_json_schema_version}')
        )

    @deferrable
    def config_sanity_check(self, trivy_output_json):
        if 'Results' not in trivy_output_json:
            return True

        results = trivy_output_json['Results']

        num_fails = 0
        for re in results:
            if not 'MisconfSummary' in re:
                continue

            summary = re['MisconfSummary']
            if 'Failures' in summary:
                fail = summary['Failures']
                num_fails += int(fail)

        return sn.assert_eq(num_fails, 0,
                            msg=f'Found {num_fails} misconfiguration(s). '
                            f'More information can be found in the '
                            f'{self.trivy_output_file} file')

    @deferrable
    def vuln_sanity_check(self, trivy_output_json):
        if 'Results' not in trivy_output_json:
            return True

        results = trivy_output_json['Results']
        severity = dict()
        for re in results:
            if 'Vulnerabilities' not in re:
                continue

            vulns = re['Vulnerabilities']
            for vul in vulns:
                if 'Severity' in vul:
                    sev = vul['Severity']
                    if sev in severity:
                        severity[sev] += 1
                    else:
                        severity[sev] = 1

        return sn.assert_eq(sn.sum(severity.values()), 0,
                            msg=f'Found {sn.sum(severity.values())} '
                                f'vulnerabilities(s). The statistics is '
                                f'{severity}. More information can be found '
                                f'in the {self.trivy_output_file} file')

    @deferrable
    def secret_sanity_check(self, trivy_output_json):
        if 'Results' not in trivy_output_json:
            return True

        results = trivy_output_json['Results']
        severity = dict()
        for re in results:
            if 'Secrets' not in re:
                continue

            secrets = re['Secrets']
            for secret in secrets:
                if 'Severity' in secret:
                    sev = secret['Severity']
                    if sev in severity:
                        severity[sev] += 1
                    else:
                        severity[sev] = 1

        return sn.assert_eq(sn.sum(severity.values()), 0,
                            msg=f'Found {sn.sum(severity.values())} '
                                f'exposed secrets(s). The statistics is '
                                f'{severity}. More information can be found in '
                                f'the {self.trivy_output_file} file')

    @sanity_function
    def assert_scan_results(self):
        '''Assert that we capture all failures.'''

        fn_name = f'{self.scanner_type}_sanity_check'
        fn_assertion = getattr(self, fn_name, None)
        trivy_output_json = self.load_json_output(self.trivy_output_file)

        skip_me = sn.extractall(r'no such file or directory', self.stderr)
        self.skip_if(skip_me,
                     msg=f'Skipping test because {self.folders} does not exist')

        return sn.all([
            sn.assert_found(f'{self.scanner_name} scanning is enabled',
                            self.stderr),
            sn.assert_not_found(f'permission denied', self.stderr),
            sn.assert_not_found(f'scan error', self.stderr),
            sn.assert_not_found(f'scan failed', self.stderr),
            self.assert_supported_schemaversion(trivy_output_json),
            fn_assertion(trivy_output_json),
        ])
