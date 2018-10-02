#
# module to test reframe's loader with relative imports
#
from . import invalid_iterable
from .. import hellocheck 
from .invalid_iterable import _get_checks
from ..hellocheck import HelloTest 
