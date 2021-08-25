# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import copy
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
    verbosity = 'v' * int(config.get('general/0/verbose'))

    # Collect the generate CI options
    before_script = config.get('generate-ci/0/before_script')
    after_script = config.get('generate-ci/0/after_script')
    artifacts = config.get('generate-ci/0/artifacts')
    artifacts_expiry = config.get('generate-ci/0/artifacts_expiry')

    # Need to append prefix to artifacts
    if artifacts:
        artifacts = [os.path.join(prefix, a) if a in [
            'stage', 'output'] else a for a in artifacts]
    else:
        artifacts = []

    def rfm_command(testcase):
        if config.filename != '<builtin>':
            config_opt = f'-C {config.filename}'
        else:
            config_opt = ''

        if testcase.level:
            restore_files = ','.join(
                f'{t.check.name}-report.json' for t in tc.deps
            )
        else:
            restore_files = None

        return ' '.join([
            program,
            f'--prefix={prefix}', config_opt,
            f'{" ".join("-c " + c for c in checkpath)}',
            f'-R' if recurse else '',
            f'--report-file={testcase.check.name}-report.json',
            f'--report-junit={testcase.check.name}-report.xml',
            f'--restore-session={restore_files}' if restore_files else '',
            f'{"".join("-" + verbosity)}' if verbosity else '',
            '-n', f'{testcase.check.name}$', '-r',  # regex $ to indicate end
        ])

    max_level = 0   # We need the maximum level to generate the stages section
    art_folder = 'rfm-artifacts'  # Folder where we place all artifacts

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
        json = {
            'image': image_name,
            **json,
        }

    for tc in testcases:
        json[f'{tc.check.name}'] = {
            'stage': f'rfm-stage-{tc.level}',
            'before_script': copy.deepcopy(before_script),
            'script': [rfm_command(tc)],
            'after_script': copy.deepcopy(after_script),
            'artifacts': {
                'paths': [f'{tc.check.name}-report.json',
                          f'{tc.check.name}-report.xml'] + artifacts,
                'expire_in': artifacts_expiry,
            },
            'needs': [t.check.name for t in tc.deps]
        }
        max_level = max(max_level, tc.level)

    # Add a last job that gathers all artifacts from precedent jobs
    json['GatherArtifacts'] = {
        'stage': f'rfm-stage-{max_level + 1}',
        'script': [f'echo \"Gathering artifacts from all jobs into '
                   f'{art_folder} folder\"',
                   f'mkdir -p {art_folder}',
                   f'cp -r {" ".join(a for a in artifacts)} *.json *.xml '
                   f'{art_folder}'],
        'needs': [tc.check.name for tc in testcases],
        'artifacts': {
            'paths': [art_folder],
            'expire_in': artifacts_expiry,
        },
    }

    # We incremented one stage by above gather artifacts job
    max_level += 1

    json['stages'] = [f'rfm-stage-{m}' for m in range(max_level+1)]
    return json


def emit_pipeline(fp, testcases, backend='gitlab'):
    if backend != 'gitlab':
        raise errors.ReframeError(f'unknown CI backend {backend!r}')

    yaml.dump(_emit_gitlab_pipeline(testcases), stream=fp,
              indent=2, sort_keys=False, width=sys.maxsize)
