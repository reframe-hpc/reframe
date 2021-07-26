import re
import reframe as rfm

__all__ = ['define_reference']

def define_reference(test: rfm.RegressionTest):
    test.reference_value = test.references[test.benchmark][0]
    test.reference_difference = test.references[test.benchmark][1]
