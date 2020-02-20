# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import reframe.utility.color as color


class TestColors(unittest.TestCase):
    def test_color_rgb(self):
        c = color.ColorRGB(128, 0, 34)
        self.assertEqual(128, c.r)
        self.assertEqual(0, c.g)
        self.assertEqual(34, c.b)

        self.assertRaises(ValueError, color.ColorRGB, -1, 0, 34)
        self.assertRaises(ValueError, color.ColorRGB, 0, -1, 34)
        self.assertRaises(ValueError, color.ColorRGB, 0, 28, -1)

    def test_colorize(self):
        s = color.colorize('hello', color.RED, palette='ANSI')
        self.assertIn('\033', s)
        self.assertIn('[3', s)
        self.assertIn('1m', s)

        with self.assertRaises(ValueError):
            color.colorize('hello', color.RED, palette='FOO')

        with self.assertRaises(ValueError):
            color.colorize('hello', color.ColorRGB(128, 0, 34), palette='ANSI')
