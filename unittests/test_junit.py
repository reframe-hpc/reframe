# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import sys

from lxml import etree
from lxml.doctestcompare import norm_whitespace
from reframe.frontend.statistics import junit_lxml

# https://github.com/SuminAndrew/lxml-asserts/blob/master/LICENSE


def raise_exc_info(exc_info):
    raise exc_info[1].with_traceback(exc_info[2])


def _describe_element(elem):
    return elem.getroottree().getpath(elem)


def _xml_compare_text(t1, t2, strip):
    t1 = t1 or ''
    t2 = t2 or ''

    if strip:
        t1 = norm_whitespace(t1).strip()
        t2 = norm_whitespace(t2).strip()

    return t1 == t2


def _assert_tag_and_attributes_are_equal(xml1, xml2, can_extend=False):
    if xml1.tag != xml2.tag:
        raise AssertionError(
            u"Tags do not match: {tag1} != {tag2}".format(
                tag1=_describe_element(xml1), tag2=_describe_element(xml2)
            )
        )

    added_attributes = set(xml2.attrib).difference(xml1.attrib)
    missing_attributes = set(xml1.attrib).difference(xml2.attrib)

    if missing_attributes:
        raise AssertionError(
            u"Second xml misses attributes: {path}/({attributes})".format(
                path=_describe_element(xml2), attributes=','.join(
                    missing_attributes)
            )
        )

    if not can_extend and added_attributes:
        raise AssertionError(
            (u"Second xml has additional attributes: "
             u"{path}/({attributes})".format(
                path=_describe_element(xml2),
                attributes=','.join(added_attributes)))
        )

    for attrib in xml1.attrib:
        if not _xml_compare_text(xml1.attrib[attrib], xml2.attrib[attrib],
                                 False):
            raise AssertionError(
                (u"Attribute values are not equal: {path}/{attribute}"
                 u"['{v1}' != '{v2}']".format(
                     path=_describe_element(xml1),
                     attribute=attrib,
                     v1=xml1.attrib[attrib],
                     v2=xml2.attrib[attrib])))

    if not _xml_compare_text(xml1.text, xml2.text, True):
        raise AssertionError(
            u"Tags text differs: {path}['{t1}' != '{t2}']".format(
                path=_describe_element(xml1), t1=xml1.text, t2=xml2.text
            )
        )

    if not _xml_compare_text(xml1.tail, xml2.tail, True):
        raise AssertionError(
            u"Tags tail differs: {path}['{t1}' != '{t2}']".format(
                path=_describe_element(xml1), t1=xml1.tail, t2=xml2.tail
            )
        )


def _assert_xml_docs_are_equal(xml1, xml2, check_tags_order=False):
    _assert_tag_and_attributes_are_equal(xml1, xml2)

    children1 = list(xml1)
    children2 = list(xml2)

    if len(children1) != len(children2):
        raise AssertionError(
            (u"Children are not equal: "
             u"{len1} children != {len2} children]".format(
                 path=_describe_element(xml1), len1=len(children1),
                 len2=len(children2))))

        raise AssertionError(
            (u"Children are not equal: "
             u"{path}[{len1} children != {len2} children]".format(
                 path=_describe_element(xml1), len1=len(children1),
                 len2=len(children2))))

    if check_tags_order:
        for c1, c2 in zip(children1, children2):
            _assert_xml_docs_are_equal(c1, c2, True)

    else:
        children1 = set(children1)
        children2 = set(children2)

        for c1 in children1:
            c1_match = None

            for c2 in children2:
                try:
                    _assert_xml_docs_are_equal(c1, c2, False)
                except AssertionError:
                    pass
                else:
                    c1_match = c2
                    break

            if c1_match is None:
                raise AssertionError(
                    u"No equal child found in second xml: {path}".format(
                        path=_describe_element(c1)
                    )
                )

            children2.remove(c1_match)


def _assert_xml_compare(cmp_func, xml1, xml2, **kwargs):
    if not isinstance(xml1, etree._Element):
        xml1 = etree.fromstring(xml1)

    if not isinstance(xml2, etree._Element):
        xml2 = etree.fromstring(xml2)

    cmp_func(xml1, xml2, **kwargs)


def assert_xml_equal(first, second, check_tags_order=False):
    _assert_xml_compare(
        _assert_xml_docs_are_equal, first, second,
        check_tags_order=check_tags_order
    )


def _generate_json():
    json_rpt = {
        'session_info': {
            'num_failures': '1',
            'num_cases': '2',
            'time_elapsed': '0.445',
            'hostname': 'dom101',
        },
        'runs': [
            {
                'testcases': [
                    {
                        'name': 'P100_Test',
                        'system': 'dom:mc',
                        'time_total': '0.179',
                        'filename': 'test.py',
                        'environment': 'PrgEnv-cray',
                        'result': 'success',
                    },
                    {
                        'name': 'V100_Test',
                        'system': 'dom:mc',
                        'time_total': '0.266',
                        'filename': 'test.py',
                        'environment': 'PrgEnv-cray',
                        'result': 'failure',
                        'fail_phase': 'sanity',
                        'fail_reason': (
                            "sanity error: pattern 'x' not found in "
                            "'rfm_V100_Test_1_job.out'"
                        ),
                    },
                ],
            }
        ],
    }
    return json_rpt


def assertXmlEqual(self, first, second, check_tags_order=False, msg=None):
    '''
    Assert that two xml documents are equal.
    :param first: first etree object or xml string
    :param second: second etree object or xml string
    :param check_tags_order: if False, the order of children is ignored
    :param msg: custom error message
    :return: raises failureException if xml documents are not equal
    '''
    if msg is None:
        msg = u'XML documents are not equal'

    try:
        assert_xml_equal(first, second, check_tags_order)
    except AssertionError as e:
        raise_exc_info(
            (
                self.failureException,
                self.failureException(u"{} — {}".format(msg, unicode_type(e))),
                sys.exc_info()[2],
            )
        )


def test_xmlreport():
    # <?xml version='1.0' encoding='utf8'?>
    reference_tree_str = """
        <testsuites>
          <testsuite name='rfm' errors='0' failures='1' tests='2' time='0.445'
                     hostname='dom101'>
            <testcase classname='test.py' name='P100_Test[dom:mc, PrgEnv-cray]'
                     time='0.179'/>
            <testcase classname='test.py'
                     name='V100_Test_1[dom:mc, PrgEnv-cray]' time='0.266'>
              <failure type='sanity'>sanity error: pattern 'x' not found in
                                     'rfm_V100_Test_1_job.out'</failure>
            </testcase>
          </testsuite>
        </testsuites>

        """.strip()
    reference_tree = etree.fromstring(reference_tree_str)
    json_report = _generate_json()
    rfm_tree = etree.fromstring(junit_lxml(json_report))
    # debug with: print(etree.tostring(rfm_tree).decode('utf-8'))
    msg = u'XML documents are not equal'
    try:
        assert_xml_equal(rfm_tree, reference_tree, check_tags_order=False)
    except AssertionError as e:
        print(f'___{e}___')
