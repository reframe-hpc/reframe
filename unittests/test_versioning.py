# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import reframe.utility.versioning as versioning


def test_version_validation():
    conditions = [versioning.VersionValidator('<=1.0.0'),
                  versioning.VersionValidator('2.0.0..2.5.0'),
                  versioning.VersionValidator('3.0.0')]

    assert all([any(c.validate('0.1.0') for c in conditions),
                any(c.validate('2.0.0') for c in conditions),
                any(c.validate('2.2.0') for c in conditions),
                any(c.validate('2.5.0') for c in conditions),
                any(c.validate('3.0.0') for c in conditions),
                not any(c.validate('3.1.0') for c in conditions)])
    with pytest.raises(ValueError):
        versioning.VersionValidator('2.0.0..')

    with pytest.raises(ValueError):
        versioning.VersionValidator('..2.0.0')

    with pytest.raises(ValueError):
        versioning.VersionValidator('1.0.0..2.0.0..3.0.0')

    with pytest.raises(ValueError):
        versioning.VersionValidator('=>2.0.0')

    with pytest.raises(ValueError):
        versioning.VersionValidator('2.0.0>')

    with pytest.raises(ValueError):
        versioning.VersionValidator('2.0.0>1.0.0')

    with pytest.raises(ValueError):
        versioning.VersionValidator('=>')

    with pytest.raises(ValueError):
        versioning.VersionValidator('>1')
