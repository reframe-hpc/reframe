#!/bin/sh
#
# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Utility script for a multi-arch installations of reframe in the same file
# system
#

# Colors
bold=$(tput bold)
red=$(tput setaf 1)
blue=$(tput setaf 4)
white=$(tput setaf 7)
reset=$(tput sgr0)


usage() {
    printf "${bold}Usage: %s [-h] [--prefix DIR] [install|uninstall]\n" "$0${reset}"
    exit 1
}

# Default values
_prefix="$HOME/.local"

action=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --prefix)
            if [ -n "$2" ] && [ "${2#-}" = "$2" ]; then
                _prefix="$2"
                shift 2
            else
                printf "Error: %s requires an argument.\n" "$1" >&2
                exit 1
            fi
            ;;
        --prefix=*)
            _prefix="${1#*=}"
            shift 1
            ;;
        --)
            shift
            break
            ;;
        -*)
            printf "error: unknown option '%s'\n" "$1" >&2
            usage
            ;;
        "install")
            action="install"
            shift
            ;;
        "uninstall")
            action="uninstall"
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [ -z $action ]; then
    echo "${bold}${red}error: no action specified${reset}"
    usage
    exit 1
fi

_prefix=${_prefix}/$(uname -m)
export UV_TOOL_BIN_DIR=${_prefix}/bin
export UV_TOOL_DIR=${_prefix}/share/uv/tools

rfm_srcdir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
case $action in
    "install")
        uv tool install $rfm_srcdir ;;
    "uninstall")
        uv tool uninstall reframe-hpc
        exit 0 ;;
esac

case "${SHELL}" in
    *bin/fish)
        cat <<EOF
${bold}Add the following lines to \`${blue}$HOME/.config/fish/config.fish${white}\`:
fish_add_path ${UV_TOOL_BIN_DIR}
set -apgx MANPATH ${UV_TOOL_DIR}/reframe-hpc/share/man ""
source ${UV_TOOL_DIR}/reframe-hpc/share/fish/vendor_completions.d/reframe.fish${reset}
EOF
        ;;
    *bin/bash|*bin/zsh)
        cat <<EOF

${bold}Add the following lines to \`${blue}$HOME/.profile${white}\`:
export PATH=${UV_TOOL_BIN_DIR}:$${PATH}
export MANPATH=${UV_TOOL_DIR}/reframe-hpc/share/man:$${MANPATH}:
source ${UV_TOOL_DIR}/reframe-hpc/share/bash-completion/completions/reframe${reset}
EOF
        ;;
    *)
        echo "${bold}${red}error${reset}: unknown shell \`${blue}$SHELL${reset}\`" ;;
esac