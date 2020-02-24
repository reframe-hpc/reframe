# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys


VERSION = '3.0-dev1'
INSTALL_PREFIX = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MIN_PYTHON_VERSION = (3, 5, 0)

# Check python version
if sys.version_info[:3] < MIN_PYTHON_VERSION:
    sys.stderr.write('Unsupported Python version: '
                     'Python >= %d.%d.%d is required\n' % MIN_PYTHON_VERSION)
    sys.exit(1)


# Import important names for user tests
from reframe.core.pipeline import *     # noqa: F401, F403
from reframe.core.decorators import *   # noqa: F401, F403
