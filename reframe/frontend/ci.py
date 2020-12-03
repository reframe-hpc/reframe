# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import yaml

import reframe
import reframe.core.exceptions as errors
import reframe.core.runtime as runtime


def _generate_gitlab_pipeline(testcases):
    rt = runtime.runtime()

    rfm_exec = f'{reframe.INSTALL_PREFIX}/bin/reframe '
    rfm_prefix = '--prefix rfm_testcases_stage_dir'
    load_path='-c '.join(rt.site_config.get('general/0/check_search_path'))
    recurse='-R' if rt.site_config.get('general/0/check_search_recursive') else ''
    report_file = 'rfm_report.json'

    # getting the max level
    max_level = 0
    for test in testcases:
        max_level = max(max_level, test.level)

    pipeline_info = {}
    for tc in testcases:
        # when restoring tests in stages we need to be able to load them
        restore_opt = f'--restore-session={report_file}' if tc.level != max_level else ''
        test_file = inspect.getfile(type(tc.check))
        pipeline_info[f'{tc.check.name}'] = {
            'stage' : f'rfm-stage-{max_level - tc.level}',
            'script' : [
                f'{rfm_exec} {rfm_prefix} -C {rt.site_config.filename} -c {load_path} {recurse} -n {tc.check.name} -r --report-file {report_file} {restore_opt}'
            ],
            'artifacts' : {
                'paths' : 'rfm_testcases_stage_dir'
            },
            'needs' : [t.check.name for t in tc.deps]
        }
        max_level = max(max_level, tc.level)

    stages = {
        'stages': [f'rfm-stage-{m}' for m in range(max_level+1)]
    }

    return stages, pipeline_info


def generate_ci_file(filename, stages, pipeline_info):
    with open(filename, 'w') as pipeline_file:
        for entry in yaml.safe_dump(stages, indent=2).split('\n'):
            pipeline_file.write(f'{entry}\n')

        for entry in yaml.safe_dump(pipeline_info, indent=2).split('\n'):
            pipeline_file.write(f'{entry}\n')


def generate_ci_pipeline(filename, testcases, backend='gitlab'):
    if backend != 'gitlab':
        raise errors.ReframeError(f'unknown CI backend {backend!r}')

    stages, pipeline_info = _generate_gitlab_pipeline(testcases)

    generate_ci_file(filename, stages, pipeline_info)
