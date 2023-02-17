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
from reframe.frontend.loader import RegressionCheckLoader


def test_ci_gitlab_pipeline():
    loader = RegressionCheckLoader([
        'unittests/resources/checks_unlisted/deps_complex.py'
    ])
    cases = dependencies.toposort(
        dependencies.build_deps(
            executors.generate_testcases(loader.load_all())
        )[0]
    )
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
