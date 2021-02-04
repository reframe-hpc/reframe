# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import io
import requests

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
        yaml = fp.getvalue()

    response = requests.post('https://gitlab.com/api/v4/ci/lint',
                             data={'content': {yaml}})
    assert response.ok
    assert response.json()['status'] == 'valid'
