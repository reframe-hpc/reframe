# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import yaml

import reframe.core.exceptions as errors
import reframe.core.runtime as runtime


def _emit_gitlab_pipeline(testcases, child_pipeline_opts):
    config = runtime.runtime().site_config

    # Collect the necessary ReFrame invariants
    program = 'reframe'
    prefix = 'rfm-stage/${CI_COMMIT_SHORT_SHA}'
    checkpath = config.get('general/0/check_search_path')
    recurse = config.get('general/0/check_search_recursive')
    verbosity = 'v' * config.get('general/0/verbose')

    def rfm_command(testcase):
        # Ignore the first argument, it should be '<builtin>'
        config_opt = ' '.join([f'-C {arg}' for arg in config.sources[1:]])

        report_file = f'{testcase.check.unique_name}-report.json'
        if testcase.level:
            restore_files = ','.join(
                f'{t.check.unique_name}-report.json' for t in tc.deps
            )
        else:
            restore_files = None

        return ' '.join([
            program,
            f'--prefix={prefix}', config_opt,
            f'{" ".join("-c " + c for c in checkpath)}',
            f'-R' if recurse else '',
            f'--report-file={report_file}',
            f'--restore-session={restore_files}' if restore_files else '',
            f'--report-junit={testcase.check.unique_name}-report.xml',
            f'{"".join("-" + verbosity)}' if verbosity else '',
            '-n', f"'^{testcase.check.unique_name}$'", '-r',
            *child_pipeline_opts
        ])

    max_level = 0   # We need the maximum level to generate the stages section
    json = {
        'cache': {
            'key': '${CI_COMMIT_REF_SLUG}',
            'paths': ['rfm-stage/${CI_COMMIT_SHORT_SHA}']
        },
        'stages': []
    }

    # Name of the image used for CI. If user does not explicitly provide
    # image keyword on the top of CI script, this variable does not exist
    image_name = os.getenv('CI_JOB_IMAGE')
    if image_name:
        json['image'] = image_name

    for tc in testcases:
        json[f'{tc.check.unique_name}'] = {
            'stage': f'rfm-stage-{tc.level}',
            'script': [rfm_command(tc)],
            'artifacts': {
                'paths': [f'{tc.check.unique_name}-report.json']
            },
            'needs': [t.check.unique_name for t in tc.deps]
        }
        max_level = max(max_level, tc.level)

    json['stages'] = [f'rfm-stage-{m}' for m in range(max_level+1)]
    return json


def emit_pipeline(fp, testcases, child_pipeline_opts=None, backend='gitlab'):
    if backend != 'gitlab':
        raise errors.ReframeError(f'unknown CI backend {backend!r}')

    child_pipeline_opts = child_pipeline_opts or []
    yaml.dump(_emit_gitlab_pipeline(testcases, child_pipeline_opts), stream=fp,
              indent=2, sort_keys=False, width=sys.maxsize)
