#
# Sanity checks and modules environment setup
#

import os
import re
import subprocess
import sys

required_version = (3, 5, 0)

# Check python version
if sys.version_info[:3] < required_version:
    sys.stderr.write('Unsupported Python version: '
                     'Python >= %d.%d.%d is required\n' % required_version)
    sys.exit(1)
