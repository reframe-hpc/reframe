import reframe.core.runtime as rt
import reframe.utility.sanity as util

def have_name(names):
    def _fn(c):
        return c.name in names

    return _fn

def have_not_name(names):
    def _fn(c):
        return not have_name(names)(c)

    return _fn

def have_tag(tags):
    def _fn(c):
        return (set(tags)).issubset(c.tags)

    return _fn

def have_prgenv(prgenv):
    def _fn(c):
        if prgenv:
            return util.allx(c.supports_environ(e) for e in prgenv)
        else:
            return bool(c.valid_prog_environs)

    return _fn

def have_system():
    def _fn(c):
        return any([c.supports_system(s.fullname)
                    for s in rt.runtime().system.partitions])

    return _fn

def is_gpu_only():
    def _fn(c):
        return c.num_gpus_per_node > 0

    return _fn

def is_cpu_only():
    def _fn(c):
        return c.num_gpus_per_node == 0
    
    return _fn
