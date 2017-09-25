#!/bin/bash

##############################################################################
#
#
#                                SCRIPT VARIABLES
#
#
##############################################################################
scriptname=`basename $0`
CI_FOLDER=""
TERM="${TERM:-xterm}"
PROFILE=""
MODULEUSE=""

#
# This function prints the script usage form
#

CI_EXITCODE=0

swap_files()
{
    if [ $# -lt 2 ]; then
       echo "too few arguments to swap_files()" 1>&2
       exit 1
    fi

    tmp="${1}_save"
    cp $1 $tmp
    cp $2 $1
    cp $tmp $2
    /bin/rm $tmp
}

usage()
{
    cat <<EOF
Usage: $(tput setaf 1)$scriptname$(tput sgr0) $(tput setaf 3)[OPTIONS]$(tput sgr0) $(tput setaf 2)-f <regression-folder>$(tput sgr0)

    $(tput setaf 3)OPTIONS:$(tput sgr0)

    $(tput setaf 3)-f | --folder$(tput sgr0) $(tput setaf 1)DIR$(tput sgr0)        ci folder, e.g. reframe-ci
    $(tput setaf 3)-i | --invocation$(tput sgr0) $(tput setaf 1)ARGS$(tput sgr0)   invocation for modified user checks. Multiple \`-i' options are multiple invocations
    $(tput setaf 3)-l | --load-profile$(tput sgr0) $(tput setaf 1)ARGS$(tput sgr0) sources the given file before any execution of commands
    $(tput setaf 3)-m | --module-use$(tput sgr0) $(tput setaf 1)ARGS$(tput sgr0)   executes module use of the give folder before loading the regression
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


run_user_checks()
{
    cmd="./bin/reframe --exec-policy=async  -r -t production $@"
    echo "Running user checks with \`$cmd'"
    checked_exec $cmd
}

run_serial_user_checks()
{
    cmd="./bin/reframe --exec-policy=serial  -r -t production-serial $@"
    echo "Running user checks with \`$cmd'"
    checked_exec $cmd
}


##############################################################################
#
#
#                              MAIN SCRIPT
#
#
##############################################################################

#
# Getting the machine name from the cmd line arguments
#

#
# GNU Linux version
#
shortopts="h,f:,i:,l:,m:"
longopts="help,folder:,invocation:,load-profile:,module-use:"

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

# Performing the unittests
echo "=================="
echo "Running unit tests"
echo "=================="
checked_exec ./test_reframe.py -v

if [ $CI_EXITCODE -eq 0 ]; then
    # Run unit tests for the public release
    swap_files reframe/settings.py reframe/settings.public.py

    echo "================================="
    echo "Running public release unit tests"
    echo "================================="
    checked_exec ./test_reframe.py -v

    swap_files reframe/settings.public.py reframe/settings.py
fi

# FIXME: Temporary workaround for the PE upgrade on Daint
if [[ $(hostname) == daint* ]]; then
    # Do not test modfied tests on Daint
    exit $CI_EXITCODE
fi


# Find modified or added user checks
userchecks=( $(git log --name-status --oneline --no-merges -1 | \
               grep -e '^[AM][[:space:]]*checks/.*\.py$' | \
               awk '{ print $2 }') )


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

exit $CI_EXITCODE
