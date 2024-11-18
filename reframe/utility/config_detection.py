# Copyright 2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

containers_detect_bash = '''
# List of containers to check
CONTAINERS=(
    "Sarus:sarus"
    "Apptainer:apptainer"
    "Docker:docker"
    "Singularity:singularity"
    "Shifter:shifter"
)

# Array to hold installed containers
installed=()

# Function to check for module existence (with lmod)
check_module_spider() {
    output=$(module spider "$1" 2>&1)
    if echo $output | grep -q "error"; then
        return 1
    else
        return 0
    fi
}

# Function to check for module existence (with tmod)
check_module_avail() {
    output=$(module avail "$1" 2>&1)
    if echo $output | grep -q "$1"; then
        return 0
    else
        return 1
    fi
}

check_lmod() {
    if [[ -n "$LMOD_CMD" ]]; then
        return 0
    else
        return 1
    fi
}

check_tmod() {
    if [[ -n "modulecmd -V" ]]; then
        return 0
    else
        return 1
    fi
}

# Check each container command
for container in "${CONTAINERS[@]}"; do
    IFS=":" read -r name cmd <<< "$container"

    # Check if the command exists via 'which'
    found_via_command=false
    found_via_module=false

    if which "$cmd" > /dev/null 2>&1; then
        found_via_command=true
    fi

    if check_lmod; then
        # Check if it is available as a module, regardless of 'which' result
        if check_module_spider "$cmd"; then
            output=$(module spider "$cmd" 2>&1)
            modules_load=$(echo $output | grep -oP '\
                (?<=available to load.).*?(?= Help)')
            found_via_module=true
        fi
    fi

    if check_tmod; then
        # Check if it is available as a module, regardless of 'which' result
        if check_module_avail "$cmd"; then
            output=$(module avail "$cmd" 2>&1)
            modules_load=""
            found_via_module=true
        fi
    fi

    # Determine the status of the container
    if $found_via_command && $found_via_module; then
        installed+=("$name modules: $modules_load")
    elif $found_via_command; then
        installed+=("$name")
    elif $found_via_module; then
        installed+=("$name modules: $modules_load")
    else
        echo "$name is not installed."
    fi
done

# Output installed containers
echo "Installed containers:"
for name in "${installed[@]}"; do
    echo "$name"
done
'''
