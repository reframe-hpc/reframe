# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.utility.color as color


def test_color_rgb():
    c = color.ColorRGB(128, 0, 34)
    assert 128 == c.r
    assert 0 == c.g
    assert 34 == c.b
    with pytest.raises(ValueError):
        color.ColorRGB(-1, 0, 34)

    with pytest.raises(ValueError):
        color.ColorRGB(0, -1, 34)

    with pytest.raises(ValueError):
        color.ColorRGB(0, 28, -1)


def test_colorize():
    s = color.colorize('hello', color.RED, palette='ANSI')
    assert '\033' in s
    assert '[3' in s
    assert '1m' in s
    with pytest.raises(ValueError):
        color.colorize('hello', color.RED, palette='FOO')

    with pytest.raises(ValueError):
        color.colorize('hello', color.ColorRGB(128, 0, 34), palette='ANSI')
