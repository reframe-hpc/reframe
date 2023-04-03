# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import io
import jsonschema
import pytest
import requests
import yaml

import reframe.frontend.ci as ci
import reframe.frontend.dependencies as dependencies
import reframe.frontend.executors as executors
from reframe.core.exceptions import ReframeError
from reframe.frontend.loader import RegressionCheckLoader


def _generate_test_cases(checks):
    return dependencies.toposort(
        dependencies.build_deps(executors.generate_testcases(checks))[0]
    )


@pytest.fixture
def hello_test():
    from unittests.resources.checks.hellocheck import HelloTest
    return HelloTest()


def test_ci_gitlab_pipeline():
    loader = RegressionCheckLoader([
        'unittests/resources/checks_unlisted/deps_complex.py'
    ])
    cases = _generate_test_cases(loader.load_all())
    with io.StringIO() as fp:
        ci.emit_pipeline(fp, cases)
        pipeline = fp.getvalue()

    # Fetch the latest Gitlab CI JSON schema
    try:
        response = requests.get(
            'https://gitlab.com/gitlab-org/gitlab/-/raw/master/app/assets/javascripts/editor/schema/ci.json'    # noqa: E501
        )
    except requests.exceptions.ConnectionError as e:
        pytest.skip(f'could not reach URL: {e}')
    else:
        assert response.ok

    schema = response.json()
    jsonschema.validate(yaml.safe_load(pipeline), schema)


def test_ci_gitlab_ci_extras(hello_test):
    hello_test.ci_extras = {
        'gitlab': {
            'before_script': ['touch foo.txt'],
            'after_script': ['echo done'],
            'artifacts': {
                'paths': ['foo.txt']
            },
            'only': {
                'changes': ['src/foo.c']
            }
        }
    }
    cases = _generate_test_cases([hello_test])
    with io.StringIO() as fp:
        ci.emit_pipeline(fp, cases)
        pipeline = fp.getvalue()

    pipeline_json = yaml.safe_load(pipeline)['HelloTest']
    assert pipeline_json['before_script'] == ['touch foo.txt']
    assert pipeline_json['after_script'] == ['echo done']
    assert pipeline_json['artifacts'] == {
        'paths': ['HelloTest-report.json', 'foo.txt']
    }
    assert pipeline_json['only'] == {'changes': ['src/foo.c']}


def test_ci_gitlab_ci_extras_invalid(hello_test):
    with pytest.raises(TypeError):
        hello_test.ci_extras = {
            'before_script': ['touch foo.txt']
        }

    with pytest.raises(TypeError):
        hello_test.ci_extras = {
            'foolab': {
                'before_script': ['touch foo.txt']
            }
        }

    # Check invalid keywords
    for kwd in ('stage', 'script', 'needs'):
        hello_test.ci_extras = {
            'gitlab': {kwd: 'something'}
        }
        cases = _generate_test_cases([hello_test])
        with io.StringIO() as fp:
            with pytest.raises(ReframeError,
                               match=rf'invalid keyword found: {kwd!r}'):
                ci.emit_pipeline(fp, cases)
