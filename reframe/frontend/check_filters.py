import re

import reframe.core.runtime as rt
import reframe.utility.sanity as util


def have_name(names):
    def _fn(c):
        return c.name in names

    return _fn


def have_name_regexp(regexp_names):
    def _fn(c):
        for p in regexp_names:
            if p.search(c.name):
                return c.name

    return _fn


def have_not_name(names):
    def _fn(c):
        return not have_name(names)(c)

    return _fn


def have_not_name_regexp(regexp_names):
    def _fn(c):
        return not have_name_regexp(regexp_names)(c)

    return _fn


def have_tag(tags):
    def _fn(c):
        return (set(tags)).issubset(c.tags)

    return _fn


def have_tag_regexp(regexp_tags):
    def _fn(c):
        if regexp_tags:
            for p in regexp_tags:
                if(any(p.search(tag) for tag in c.tags) is False):
                    return False
        return True

    return _fn


def have_prgenv(prgenv):
    def _fn(c):
        if prgenv:
            return util.allx(c.supports_environ(e) for e in prgenv)
        else:
            return bool(c.valid_prog_environs)

    return _fn


def have_prgenv_regexp(regexp_prgenv):
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
