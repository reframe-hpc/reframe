import reframe as rfm


@rfm.simple_test
class Test1Check(rfm.RegressionTest):
    pass


@rfm.required_version()
@rfm.simple_test
class Test2Check(rfm.RegressionTest):
    pass


@rfm.simple_test
@rfm.required_version('>100.0')
class Test3Check(rfm.RegressionTest):
    pass


@rfm.required_version('1.0..2.0', '<100.0')
@rfm.simple_test
class Test4Check(rfm.RegressionTest):
    pass


@rfm.simple_test
@rfm.required_version('!=2.0')
class Test5Check(rfm.RegressionTest):
    pass
