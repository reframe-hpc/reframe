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
    "$@"
    if [ $? -ne 0 ]; then
        CI_EXITCODE=1
    fi
}

run_tutorial_checks()
{
    cmd="./bin/reframe -C tutorial/config/settings.py --exec-policy=async \
--save-log-files -r -t tutorial $@"
    echo "Running tutorial checks with \`$cmd'"
    checked_exec $cmd
}

run_user_checks()
{
    cmd="./bin/reframe -C config/cscs.py --exec-policy=async --save-log-files \
-r -t production $@"
    echo "Running user checks with \`$cmd'"
    checked_exec $cmd
}

run_serial_user_checks()
{
    cmd="./bin/reframe -C config/cscs.py --exec-policy=serial --save-log-files \
-r -t production-serial $@"
    echo "Running user checks with \`$cmd'"
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
            echo "${scriptname}: Unrecognized argument \`$1'" >&2
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

module load reframe

echo "=============="
echo "Loaded Modules"
echo "=============="
module list


cd ${CI_FOLDER}
echo "Running regression on $(hostname) in ${CI_FOLDER}"

if [ $CI_GENERIC -eq 1 ]; then
    # Run unit tests for the public release
    echo "========================================"
    echo "Running unit tests with generic settings"
    echo "========================================"
    checked_exec ./test_reframe.py
    checked_exec ! ./bin/reframe.py --system=generic -l 2>&1 | \
        grep -- '--- Logging error ---'
elif [ $CI_TUTORIAL -eq 1 ]; then
    # Run tutorial checks
    # Find modified or added tutorial checks
    tutorialchecks=( $(git diff origin/master...HEAD --name-only --oneline --no-merges | \
                       grep -e '^tutorial/(?!config/).*\.py') )

    if [ ${#tutorialchecks[@]} -ne 0 ]; then
        tutorialchecks_path=""
        for check in ${tutorialchecks[@]}; do
            tutorialchecks_path="${tutorialchecks_path} -c ${check}"
        done

        echo "========================"
        echo "Modified tutorial checks"
        echo "========================"
        echo ${tutorialchecks_path}

        for i in ${!invocations[@]}; do
            run_tutorial_checks ${tutorialchecks_path} ${invocations[i]}
        done
    fi
else
    # Performing the unittests
    echo "=================="
    echo "Running unit tests"
    echo "=================="

    checked_exec ./test_reframe.py --rfm-user-config=config/cscs.py

    echo "==================================="
    echo "Running unit tests with PBS backend"
    echo "==================================="

    if [[ $(hostname) =~ dom ]]; then
        PATH_save=$PATH
        export PATH=/apps/dom/UES/karakasv/slurm-wrappers/bin:$PATH
        checked_exec ./test_reframe.py --rfm-user-config=config/cscs-pbs.py
        export PATH=$PATH_save
    fi

    # Find modified or added user checks
    userchecks=( $(git diff origin/master...HEAD --name-only --oneline --no-merges | \
                   grep -e '^cscs-checks/.*\.py') )

    if [ ${#userchecks[@]} -ne 0 ]; then
        userchecks_path=""
        for check in ${userchecks[@]}; do
            userchecks_path="${userchecks_path} -c ${check}"
        done

        echo "===================="
        echo "Modified user checks"
        echo "===================="
        echo ${userchecks_path}

        #
        # Running the user checks
        #
        for i in ${!invocations[@]}; do
            run_user_checks ${userchecks_path} ${invocations[i]}
            run_serial_user_checks ${userchecks_path} ${invocations[i]}
        done
    fi
fi
exit $CI_EXITCODE
