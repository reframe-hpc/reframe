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
            self.foo(2)

    class _Derived_3(_Derived_1):
        pass

    class _Derived_4(_Derived_1):
        # Verify that inheritance works even if we redefine __init__()
        # completely
        def __init__(self):
            pass

    t1 = _Derived_1()
    t2 = _Derived_2()
    t3 = _Derived_3()
    t4 = _Derived_4()
    assert t1.x == 1
    assert t2.x == 2
    assert t3.x == 1
    assert t4.x == 1
