import re

import reframe.core.runtime as rt
import reframe.utility.sanity as util


def have_name(regexp_names):
    def _fn(c):
        for p in regexp_names:
            if p.search(c.name):
                return c.name

    return _fn


def have_not_name(regexp_names):
    def _fn(c):
        return not have_name(regexp_names)(c)

    return _fn


def have_tag(regexp_tags):
    def _fn(c):
        if regexp_tags:
            for p in regexp_tags:
                if(any(p.search(tag) for tag in c.tags) is False):
                    return False
        return True

    return _fn


def have_prgenv(regexp_prgenv):
    def _fn(c):
        if regexp_prgenv:
            for p in regexp_prgenv:
                if(any(p.search(prgenv) for prgenv in c.valid_prog_environs) 
                   is False):
                    return False
            return True
        else:
            return bool(c.valid_prog_environs)

    return _fn


def have_partition(partitions):
    def _fn(c):
        return any([c.supports_system(s.fullname) for s in partitions])

    return _fn


def have_gpu_only():
    def _fn(c):
        return c.num_gpus_per_node > 0

    return _fn


def have_cpu_only():
    def _fn(c):
        return c.num_gpus_per_node == 0

    return _fn
