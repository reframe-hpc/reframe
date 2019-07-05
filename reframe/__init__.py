import os
import sys


VERSION = '2.19-dev1'
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
