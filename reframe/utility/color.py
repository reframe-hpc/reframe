# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


class ColorRGB:
    def __init__(self, r, g, b):
        self.__check_rgb(r)
        self.__check_rgb(g)
        self.__check_rgb(b)
        self.__r = r
        self.__g = g
        self.__b = b

    def __check_rgb(self, x):
        if (x < 0) or x > 255:
            raise ValueError('RGB color code must be in [0,255]')

    @property
    def r(self):
        return self.__r

    @property
    def g(self):
        return self.__g

    @property
    def b(self):
        return self.__b

    def __repr__(self):
        return 'ColorRGB(%s, %s, %s)' % (self.__r, self.__g, self.__b)


# Predefined colors
BLACK   = ColorRGB(0, 0, 0)
RED     = ColorRGB(255, 0, 0)
GREEN   = ColorRGB(0, 255, 0)
YELLOW  = ColorRGB(255, 255, 0)
BLUE    = ColorRGB(0, 0, 255)
MAGENTA = ColorRGB(255, 0, 255)
CYAN    = ColorRGB(0, 255, 255)
WHITE   = ColorRGB(255, 255, 255)


class _AnsiPalette:
    '''Class for colorizing strings using ANSI meta-characters.'''

    escape_seq = '\033'
    reset_term = '[0m'

    # Escape sequences for fore/background colors
    fgcolor = '[3'
    bgcolor = '[4'

    # color values
    colors = {
        BLACK:   '0m',
        RED:     '1m',
        GREEN:   '2m',
        YELLOW:  '3m',
        BLUE:    '4m',
        MAGENTA: '5m',
        CYAN:    '6m',
        WHITE:   '7m'
    }

    def colorize(string, foreground):
        try:
            foreground = _AnsiPalette.colors[foreground]
        except KeyError:
            raise ValueError('could not find an ANSI representation '
                             'for color: %s' % foreground) from None

        return (_AnsiPalette.escape_seq +
                _AnsiPalette.fgcolor + foreground + string +
                _AnsiPalette.escape_seq + _AnsiPalette.reset_term)


def colorize(string, foreground, *, palette='ANSI'):
    '''Colorize a string.

    :arg string: The string to be colorized.
    :arg foreground: The foreground color.
    :arg palette: The palette to get colors from.
    '''
    if palette != 'ANSI':
        raise ValueError('unknown color palette: %s' % palette)

    return _AnsiPalette.colorize(string, foreground)
