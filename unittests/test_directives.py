# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.core.directives as directives


def test_directives(monkeypatch):
    monkeypatch.setattr(directives, 'NAMES', ('foo',))

    class _Base(rfm.RegressionTest):
        def _D_foo(self, x):
            self.x = x

    class _Derived_1(_Base):
        foo(1)

    class _Derived_2(_Base):
        def __init__(self):
            self.foo(1)

    t1 = _Derived_1()
    t2 = _Derived_2()
    assert t1.x == 1
    assert t2.x == 1
