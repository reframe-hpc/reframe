# Copyright 2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

nvidia_gpu_architecture = {
    "Tesla K20": "sm_35",
    "Tesla K40": "sm_35",
    "Tesla P100": "sm_60",
    "Tesla V100": "sm_70",
    "Tesla T4": "sm_75",
    "Tesla A100": "sm_80",
    "Quadro RTX 8000": "sm_75",
    "Quadro RTX 6000": "sm_75",
    "Quadro P6000": "sm_61",
    "Quadro GV100": "sm_70",
    "GeForce GTX 1080": "sm_61",
    "GeForce GTX 1080 Ti": "sm_61",
    "GeForce GTX 1070": "sm_61",
    "GeForce GTX 1060": "sm_61",
    "GeForce GTX 1050": "sm_61",
    "GeForce RTX 2060": "sm_75",
    "GeForce RTX 2070": "sm_75",
    "GeForce RTX 2080": "sm_75",
    "GeForce RTX 2080 Ti": "sm_75",
    "GeForce RTX 3060": "sm_86",
    "GeForce RTX 3070": "sm_86",
    "GeForce RTX 3080": "sm_86",
    "GeForce RTX 3090": "sm_86",
    "GeForce RTX 4060": "sm_89",
    "GeForce RTX 4070": "sm_89",
    "GeForce RTX 4080": "sm_89",
    "GeForce RTX 4090": "sm_89",
    "A100": "sm_80",
    "H100": "sm_90",
    "H200": "sm_90",
    "H100 PCIe": "sm_90",
    "H100 SXM": "sm_90",
    "Titan V": "sm_70",
    "Titan RTX": "sm_75"
}

amd_gpu_architecture = {
    # RDNA 3 Series
    "RX 7900 XTX": "RDNA 3",
    "RX 7900 XT": "RDNA 3",
    "RX 7800 XT": "RDNA 3",
    "RX 7700 XT": "RDNA 3",
    "RX 7600": "RDNA 3",

    # RDNA 2 Series
    "RX 6950 XT": "RDNA 2",
    "RX 6900 XT": "RDNA 2",
    "RX 6800 XT": "RDNA 2",
    "RX 6800": "RDNA 2",
    "RX 6750 XT": "RDNA 2",
    "RX 6700 XT": "RDNA 2",
    "RX 6700": "RDNA 2",
    "RX 6650 XT": "RDNA 2",
    "RX 6600 XT": "RDNA 2",
    "RX 6600": "RDNA 2",
    "RX 6500 XT": "RDNA 2",
    "RX 6400": "RDNA 2",

    # RDNA Series
    "RX 5700 XT": "RDNA",
    "RX 5700": "RDNA",
    "RX 5600 XT": "RDNA",
    "RX 5600": "RDNA",
    "RX 5500 XT": "RDNA",
    "RX 5500": "RDNA",
    "RX 5300": "RDNA",
    "RX 5300M": "RDNA",

    # Vega Series
    "RX Vega 64": "Vega",
    "RX Vega 56": "Vega",
    "Radeon Vega Frontier Edition": "Vega",
    "Radeon Pro Vega 64": "Vega",
    "Radeon Pro Vega 56": "Vega",

    # Vega 20 Series
    "Radeon Pro VII": "Vega 20",
    "Instinct MI50": "Vega 20",
    "Instinct MI60": "Vega 20",

    # Polaris (RX 500 Series)
    "RX 590": "Polaris (GCN 4)",
    "RX 580": "Polaris (GCN 4)",
    "RX 570": "Polaris (GCN 4)",
    "RX 560": "Polaris (GCN 4)",
    "RX 550": "Polaris (GCN 4)",

    # Polaris (RX 400 Series)
    "RX 480": "Polaris (GCN 4)",
    "RX 470": "Polaris (GCN 4)",
    "RX 460": "Polaris (GCN 4)",

    # Fury and Nano (Fiji, GCN 3)
    "R9 Fury X": "Fiji (GCN 3)",
    "R9 Fury": "Fiji (GCN 3)",
    "R9 Nano": "Fiji (GCN 3)",
    "R9 Fury Nano": "Fiji (GCN 3)",

    # R9 300 Series (GCN 2 and GCN 3)
    "R9 390X": "Hawaii (GCN 2)",
    "R9 390": "Hawaii (GCN 2)",
    "R9 380X": "Antigua (GCN 3)",
    "R9 380": "Antigua (GCN 3)",
    "R9 370X": "Trinidad (GCN 1)",
    "R9 370": "Trinidad (GCN 1)",

    # R7 300 Series (GCN 1)
    "R7 370": "Trinidad (GCN 1)",
    "R7 360": "Bonaire (GCN 1)",

    # R9 200 Series (GCN 1 and GCN 2)
    "R9 290X": "Hawaii (GCN 2)",
    "R9 290": "Hawaii (GCN 2)",
    "R9 280X": "Tahiti (GCN 1)",
    "R9 280": "Tahiti (GCN 1)",
    "R9 270X": "Curacao (GCN 1)",
    "R9 270": "Curacao (GCN 1)",
    "R9 260X": "Bonaire (GCN 1)",
    "R9 260": "Bonaire (GCN 1)",

    # R7 200 Series (GCN 1)
    "R7 265": "Pitcairn (GCN 1)",
    "R7 260X": "Bonaire (GCN 1)",
    "R7 260": "Bonaire (GCN 1)",
    "R7 250X": "Oland (GCN 1)",
    "R7 250": "Oland (GCN 1)",
    "R7 240": "Oland (GCN 1)",

    # HD 7000 Series (GCN 1)
    "HD 7970": "Tahiti (GCN 1)",
    "HD 7950": "Tahiti (GCN 1)",
    "HD 7870": "Pitcairn (GCN 1)",
    "HD 7850": "Pitcairn (GCN 1)",
    "HD 7790": "Bonaire (GCN 1)",
    "HD 7770": "Cape Verde (GCN 1)",
    "HD 7750": "Cape Verde (GCN 1)"
}

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

devices_detect_bash = '''
# Check for NVIDIA GPUs
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "Checking for NVIDIA GPUs..."
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)

    if [ -z "$gpu_info" ]; then
        echo "No NVIDIA GPU found."
    else
        echo "NVIDIA GPUs installed:"
        echo "$gpu_info"
    fi
else
    echo "No NVIDIA GPU found or nvidia-smi command is not available."
fi

# Check for AMD GPUs (if applicable)
if command -v lspci > /dev/null 2>&1; then
    echo -e "\nChecking for AMD GPUs:"
    if lspci | grep -i 'radeon'; then
        lspci | grep -i 'radeon'
    else
        echo "No AMD GPU found."
    fi
else
    echo "lspci command is not available."
fi
'''
