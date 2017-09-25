#
# reframe.utility.functions -- utility functions
#

from reframe.core.exceptions import ReframeError


def _expect_interval(val, interval, valdescr=None):
    lower, upper = interval
    if val < lower or val > upper:
        if not valdescr:
            valdescr = 'value'

        raise ReframeError('%s (%s) not in [%s,%s]' %
                           (valdescr, val, lower, upper))


def _bound(refval, thres):
    # Upper/lower bounds computation is common, since lower bounds are negative
    return (refval + abs(refval) * thres) if refval else thres


def standard_threshold(value, reference, logger=None):
    try:
        refval, thres_lower, thres_upper = reference
    except (ValueError, TypeError):
        raise ReframeError('Improper reference value')

    if logger:
        logger.info('value: %s, reference: %s' % (str(value), reference))

    # sanity checking of user input
    if refval is None:
        raise ReframeError(
            'No reference value specified for calculating tolerance')

    if thres_lower is None and thres_upper is None:
        return True

    if thres_lower is None:
        _expect_interval(thres_upper, (0, 1), 'reference upper threshold')
        return value <= _bound(refval, thres_upper)

    if thres_upper is None:
        _expect_interval(thres_lower, (-1, 0), 'reference lower threshold')
        return value >= _bound(refval, thres_lower)

    _expect_interval(thres_upper, (0, 1), 'reference upper threshold')
    _expect_interval(thres_lower, (-1, 0), 'reference lower threshold')

    return (value >= _bound(refval, thres_lower) and
            value <= _bound(refval, thres_upper))


def always_true(value, reference, **kwargs):
    return True
