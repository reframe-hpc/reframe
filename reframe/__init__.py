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


if 'MODULESHOME' not in os.environ:
    sys.stderr.write('MODULESHOME is not set. '
                     'Do you have modules framework installed? Exiting...\n')
    sys.exit(1)


MODULECMD = 'modulecmd'
MODULECMD_PYTHON = MODULECMD + ' python'
try:
    _completed = subprocess.run(args=MODULECMD_PYTHON.split(),
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
    if re.search('Unknown shell type', _completed.stderr, re.MULTILINE):
        sys.stderr.write('Python is not supported by this modules framework.\n')
        sys.exit(1)

except OSError:
    # modulecmd was not found
    sys.stderr.write("Could not run modulecmd. Tried `%s' and failed.\n" % \
                     MODULECMD_PYTHON)
    sys.exit(1)
