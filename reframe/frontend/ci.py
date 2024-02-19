# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import yaml

import reframe.core.runtime as runtime
from reframe.core.exceptions import ReframeError


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
            '-n', f'/{testcase.check.hashcode}', '-r',
            *child_pipeline_opts
        ])

    def _valid_ci_extras(extras):
        '''Validate Gitlab CI pipeline extras'''

        for kwd in ('stage', 'script', 'needs'):
            if kwd in extras:
                errmsg = f"invalid keyword found: {kwd!r}"
                if kwd == 'script':
                    errmsg += " (use 'before_script' or 'after_script')"

                raise ReframeError(f"could not validate 'ci_extras': {errmsg}")

        return extras

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
        ci_extras = _valid_ci_extras(tc.check.ci_extras.get('gitlab', {}))
        extra_artifacts = ci_extras.pop('artifacts', {})
        extra_artifact_paths = extra_artifacts.pop('paths', [])
        json[f'{tc.check.unique_name}'] = {
            'stage': f'rfm-stage-{tc.level}',
            'script': [rfm_command(tc)],
            'artifacts': {
                'paths': [f'{tc.check.unique_name}-report.json',
                          *extra_artifact_paths],
                **extra_artifacts
            },
            'needs': [t.check.unique_name for t in tc.deps],
            **ci_extras
        }
        max_level = max(max_level, tc.level)

    json['stages'] = [f'rfm-stage-{m}' for m in range(max_level+1)]
    return json


def emit_pipeline(fp, testcases, child_pipeline_opts=None, backend='gitlab'):
    if backend != 'gitlab':
        raise ReframeError(f'unknown CI backend {backend!r}')

    child_pipeline_opts = child_pipeline_opts or []
    yaml.dump(_emit_gitlab_pipeline(testcases, child_pipeline_opts), stream=fp,
              indent=2, sort_keys=False, width=sys.maxsize)
