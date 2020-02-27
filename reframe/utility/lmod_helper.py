import subprocess

class PopenCommunicate():
    def exec(self, command):
        proc=subprocess.Popen(command,
                              shell = True,
                              stderr=subprocess.STDOUT,
                              stdout=subprocess.PIPE)
        return str(proc.communicate()[0]) # None-terminated, removed None


def _all_avail_mods(thing_to_load, ml_av_output):
    all_versions = []
    look_for =  thing_to_load + "/"
    for item in ml_av_output.replace("\n", " ").split(" "):
        if item.strip().startswith(look_for):
            all_versions.append(item.strip().split("/")[1])

    return all_versions

def _loop_among_one(module_glob, pc, env_context=""):
    glob_to_load = module_glob.split()[-1]
    if not glob_to_load.endswith("/*"):
        raise Exception("Environment bit '" + module_glob +
                        "' is incorrect. Must be like 'module load gnu/*'")

    glob_to_load = glob_to_load[0:len(glob_to_load)-2]

    ml_av_out = pc.exec(env_context + "module avail " + glob_to_load)
    all_versions = _all_avail_mods(glob_to_load, ml_av_out)
    all_mods = []
    for a_version in all_versions:
        all_mods.append(module_glob.replace("*", a_version))
    return all_mods

def _expand_module_set(module_set, module_glob, pc):
    new_module_set = []
    if len(module_set) == 0:
        new_module_set = _loop_among_one(module_glob, pc)
    else:
        # combinatorial explosion of all that is found
        for item in module_set:
            for new_module in _loop_among_one(module_glob, pc, item + "; "):
                new_module_set.append(item + "; " + new_module)
    return new_module_set

def _append_to_all(module_set, a_module):
    if len(module_set) == 0:
        module_set = [a_module]
    else:
        for i, item in enumerate(module_set):
            module_set[i] = item + "; " + a_module
    return module_set

popen_comm = PopenCommunicate()
def loop_among_all(required_modules, pc=popen_comm):
    star_count = required_modules.count("*")
    if star_count == 0:
        return [required_modules]

    module_set = []
    for module_glob in required_modules.split(";"):
        if not "*" in module_glob:
            module_set = _append_to_all(module_set, module_glob.strip())
        else:
            module_set = _expand_module_set(module_set, module_glob.strip(), pc)
    return module_set
