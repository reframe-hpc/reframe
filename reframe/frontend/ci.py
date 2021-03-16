# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import yaml

import reframe.core.exceptions as errors
import reframe.core.runtime as runtime


def _emit_gitlab_pipeline(testcases):
    config = runtime.runtime().site_config

    # Collect the necessary ReFrame invariants
    program = 'reframe'
    prefix = 'rfm-stage/${CI_COMMIT_SHORT_SHA}'
    checkpath = config.get('general/0/check_search_path')
    recurse = config.get('general/0/check_search_recursive')

    def rfm_command(testcase):
        if config.filename != '<builtin>':
            config_opt = f'-C {config.filename}'
        else:
            config_opt = ''

        report_file = f'rfm-report-{testcase.level}.json'
        if testcase.level:
            restore_file = f'rfm-report-{testcase.level - 1}.json'
        else:
            restore_file = None

        return ' '.join([
            program,
            f'--prefix={prefix}', config_opt,
            f'{" ".join("-c " + c for c in checkpath)}',
            f'-R' if recurse else '',
            f'--report-file={report_file}',
            f'--restore-session={restore_file}' if restore_file else '',
            '-n', testcase.check.name, '-r'
        ])

    max_level = 0   # We need the maximum level to generate the stages section
    json = {
        'cache': {
            'key': '${CI_COMMIT_REF_SLUG}',
            'paths': ['rfm-stage/${CI_COMMIT_SHORT_SHA}']
        },
        'stages': []
    }
    for tc in testcases:
        json[f'{tc.check.name}'] = {
            'stage': f'rfm-stage-{tc.level}',
            'script': [rfm_command(tc)],
            'artifacts': {
                'paths': [f'rfm-report-{tc.level}.json']
            },
            'needs': [t.check.name for t in tc.deps]
        }
        max_level = max(max_level, tc.level)

    json['stages'] = [f'rfm-stage-{m}' for m in range(max_level+1)]
    return json


def emit_pipeline(fp, testcases, backend='gitlab'):
    if backend != 'gitlab':
        raise errors.ReframeError(f'unknown CI backend {backend!r}')

    yaml.dump(_emit_gitlab_pipeline(testcases), stream=fp,
              indent=2, sort_keys=False, width=sys.maxsize)
