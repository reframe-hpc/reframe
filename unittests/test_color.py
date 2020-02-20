import pytest
import unittest

import reframe.utility.color as color


class TestColors(unittest.TestCase):
    def test_color_rgb(self):
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

    def test_colorize(self):
        s = color.colorize('hello', color.RED, palette='ANSI')
        assert '\033' in s
        assert '[3' in s
        assert '1m' in s

        with pytest.raises(ValueError):
            color.colorize('hello', color.RED, palette='FOO')

        with pytest.raises(ValueError):
            color.colorize('hello', color.ColorRGB(128, 0, 34), palette='ANSI')
