#!/bin/bash

scriptname=`basename $0`
CI_FOLDER=""
CI_GENERIC=0
CI_TUTORIAL=0
CI_EXITCODE=0
TERM="${TERM:-xterm}"
PROFILE=""
MODULEUSE=""


#
# This function prints the script usage form
#
usage()
{
    cat <<EOF
Usage: $(tput setaf 1)$scriptname$(tput sgr0) $(tput setaf 3)[OPTIONS]$(tput sgr0) $(tput setaf 2)-f <regression-folder>$(tput sgr0)

    $(tput setaf 3)OPTIONS:$(tput sgr0)

    $(tput setaf 3)-f | --folder$(tput sgr0) $(tput setaf 1)DIR$(tput sgr0)        ci folder, e.g. reframe-ci
    $(tput setaf 3)-i | --invocation$(tput sgr0) $(tput setaf 1)ARGS$(tput sgr0)   invocation for modified user checks. Multiple \`-i' options are multiple invocations
    $(tput setaf 3)-l | --load-profile$(tput sgr0) $(tput setaf 1)ARGS$(tput sgr0) sources the given file before any execution of commands
    $(tput setaf 3)-m | --module-use$(tput sgr0) $(tput setaf 1)ARGS$(tput sgr0)   executes module use of the give folder before loading the regression
    $(tput setaf 3)-g | --generic-only$(tput sgr0)      executes unit tests using the generic configuration
    $(tput setaf 3)-t | --tutorial-only$(tput sgr0)     executes only the modified/new tutorial tests
    $(tput setaf 3)-h | --help$(tput sgr0)              prints this help and exits

EOF
} # end of usage

checked_exec()
{
    echo "[RUN] $@" && "$@"
    if [ $? -ne 0 ]; then
        CI_EXITCODE=1
    fi
}

run_tutorial_checks()
{
    cmd="./bin/reframe -C tutorial/config/settings.py \
--save-log-files -r -t tutorial $@"
    echo "[INFO] Running tutorial checks with \`$cmd'"
    checked_exec $cmd
}

run_user_checks()
{
    cmd="./bin/reframe -C config/cscs.py --save-log-files \
-r --flex-alloc-nodes=2 -t production|benchmark $@"
    echo "[INFO] Running user checks with \`$cmd'"
    checked_exec $cmd
}

### Main script ###

shortopts="h,g,t,f:,i:,l:,m:"
longopts="help,generic-only,tutorial-only,folder:,invocation:,load-profile:,module-use:"

eval set -- $(getopt -o ${shortopts} -l ${longopts} \
                     -n ${scriptname} -- "$@" 2> /dev/null)

num_invocations=0
invocations=()
while [ $# -ne 0 ]; do
    case $1 in
        -h | --help)
            usage
            exit 0 ;;
        -f | --folder)
            shift
            CI_FOLDER="$1" ;;
        -i | --invocation)
            shift

            # We need to set explicitly the array elements, in order to account
            # for whitespace characters. The alternative way of expanding the
            # array `invocations+=($1)' whould not treat whitespace correctly.
            invocations[$num_invocations]=$1
            ((num_invocations++)) ;;
        -l | --load-profile)
            shift
            PROFILE="$1" ;;
        -m | --module-use)
            shift
            MODULEUSE="$1" ;;
        -g | --generic-only)
            shift
            CI_GENERIC=1 ;;
        -t | --tutorial-only)
            shift
            CI_TUTORIAL=1 ;;
        --)
            ;;
        *)
            echo "[ERROR] ${scriptname}: Unrecognized argument \`$1'" >&2
            usage
            exit 1 ;;
    esac
    shift
done

#
# Check if package_list_file was defined
#
if [ "X${CI_FOLDER}" == "X" ]; then
    usage
    exit 1
fi

#
# Sourcing a given profile
#
if [ "X${PROFILE}" != "X" ]; then
    source ${PROFILE}
fi

#
# module use for a given folder
#
if [ "X${MODULEUSE}" != "X" ]; then
    module use ${MODULEUSE}
fi

if [[ $(hostname) =~ tsa ]]; then
    # FIXME: Temporary workaround until we have a reframe module on Tsa
    module load python
else
    module load reframe
fi

# Always install our requirements
python3 -m venv venv.unittests
source venv.unittests/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# FIXME: XALT is causing linking problems (see UES-823)
module unload xalt

echo "[INFO] Loaded Modules"
module list

cd ${CI_FOLDER}
echo "[INFO] Running unit tests on $(hostname) in ${CI_FOLDER}"

if [ $CI_GENERIC -eq 1 ]; then
    # Run unit tests for the public release
    echo "[INFO] Running unit tests with generic settings"
    checked_exec ./test_reframe.py \
                 -W=error::reframe.core.exceptions.ReframeDeprecationWarning -ra
    checked_exec ! ./bin/reframe.py --system=generic -l 2>&1 | \
        grep -- '--- Logging error ---'
elif [ $CI_TUTORIAL -eq 1 ]; then
    # Run tutorial checks
    # Find modified or added tutorial checks
    tutorialchecks=( $(git diff origin/master...HEAD --name-only --oneline --no-merges | \
                       grep -e '^tutorial/.*\.py') )

    if [ ${#tutorialchecks[@]} -ne 0 ]; then
        tutorialchecks_path=""
        for check in ${tutorialchecks[@]}; do
            tutorialchecks_path="${tutorialchecks_path} -c ${check}"
        done

        echo "[INFO] Modified tutorial checks"
        echo ${tutorialchecks_path}
        for i in ${!invocations[@]}; do
            run_tutorial_checks ${tutorialchecks_path} ${invocations[i]}
        done
    fi
else
    # Run unit tests with the scheduler backends
    tempdir=$(mktemp -d -p $SCRATCH)
    if [[ $(hostname) =~ dom ]]; then
        PATH_save=$PATH
        export PATH=/apps/dom/UES/karakasv/slurm-wrappers/bin:$PATH
        for backend in slurm pbs torque; do
            echo "[INFO] Running unit tests with ${backend}"
            checked_exec ./test_reframe.py --rfm-user-config=config/cscs-ci.py \
                         -W=error::reframe.core.exceptions.ReframeDeprecationWarning \
                         --rfm-user-system=dom:${backend} --basetemp=$tempdir -ra
        done
        export PATH=$PATH_save
    else
        echo "[INFO] Running unit tests"
        checked_exec ./test_reframe.py --rfm-user-config=config/cscs-ci.py \
                     -W=error::reframe.core.exceptions.ReframeDeprecationWarning \
                     --basetemp=$tempdir -ra
    fi

    if [ $CI_EXITCODE -eq 0 ]; then
        /bin/rm -rf $tempdir
    fi

    # Find modified or added user checks
    userchecks=( $(git diff origin/master...HEAD --name-only --oneline --no-merges | \
                   grep -e '^cscs-checks/.*\.py') )
    if [ ${#userchecks[@]} -ne 0 ]; then
        userchecks_path=""
        for check in ${userchecks[@]}; do
            userchecks_path="${userchecks_path} -c ${check}"
        done

        echo "[INFO] Modified user checks"
        echo ${userchecks_path}

        #
        # Running the user checks
        #
        for i in ${!invocations[@]}; do
            run_user_checks ${userchecks_path} ${invocations[i]}
        done
    fi
fi
deactivate
exit $CI_EXITCODE
