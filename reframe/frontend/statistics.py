import copy

class CheckFailureInfo:
    def __init__(self, check, env):
        self.check_name = check.name
        self.check_stagedir = check.stagedir
        self.prgenv = env.name


class RegressionStats:
    def __init__(self):
        self.num_checks  = 0
        self.num_fails   = 0
        self.num_cases   = 0
        self.failed_info = []


    def add_failure(self, check, env):
        self.num_fails += 1
        self.failed_info.append(CheckFailureInfo(check, env))


    def details(self):
        lines = [ '  | Summary of failed tests' ]
        for fail in self.failed_info:
            lines += [
                "    * %s failed with `%s'" % (fail.check_name, fail.prgenv),
                "          Staged in `%s'"  % fail.check_stagedir
            ]

        return '\n'.join(lines)


    def summary(self):
        return '  | Ran %d case(s) of %d supported check(s) (%d failure(s))' % \
               (self.num_cases, self.num_checks, self.num_fails)


    def __str__(self):
        return self.summary()
