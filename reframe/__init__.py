#
# Sanity checks and modules environment setup
#

import os
import re
import subprocess
import sys


VERSION = '2.9'
_required_pyver = (3, 5, 0)

# Check python version
if sys.version_info[:3] < _required_pyver:
    sys.stderr.write('Unsupported Python version: '
                     'Python >= %d.%d.%d is required\n' % _required_pyver)
    sys.exit(1)
