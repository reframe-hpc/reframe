#/usr/bin/env bash


_bin=reframe

_cli_options=$(${_bin} --help | grep -o -e ' --[a-z\-]*' -e ' -[a-zA-Z]' --)

_complete_test_name()
{
    name_line=("${COMP_WORDS[0]}")
    for i in ${!COMP_WORDS[@]};  do
        if [ "${COMP_WORDS[$i]}" == -c ] || [ "${COMP_WORDS[$i]}" == --checkpath ] ; then
            path="${COMP_WORDS[$(($i+1))]}"
            if [[ -a "$path" ]] ; then
                name_line+=("-c" "$path")
            fi
        fi

        if [ "${COMP_WORDS[$i]}" == -R ] || [ "${COMP_WORDS[$i]}" == --recursive ] ; then
            name_line+=("-R")
        fi

        if [ "${COMP_WORDS[$i]}" == -t ] || [ "${COMP_WORDS[$i]}" == --tags ]; then
            tags="${COMP_WORDS[$(($i+1))]}"
                name_line+=("-t" "$tags")
        fi

        if [ "${COMP_WORDS[$i]}" == -p ] || [ "${COMP_WORDS[$i]}" == --prgenv ]; then
            prgenv="${COMP_WORDS[$(($i+1))]}"
                name_line+=("-p" "prgenv")
        fi

        if [ "${COMP_WORDS[$i]}" == --system ]; then
            system="${COMP_WORDS[$(($i+1))]}"
                name_line+=("-p" "system")
        fi
    done

    name_line+=("-l")
    echo ${name_line[@]}
}


_cli_options_completions()
{
    local cur prev
    cur=${COMP_WORDS[COMP_CWORD]}
    prev=${COMP_WORDS[COMP_CWORD-1]}
    
    COMPREPLY=($(compgen -W "${_cli_options}" -- "${cur}"))
    
    case ${prev} in
        -n|--name)
              local test_names=$(_complete_test_name)
              test_names=$($(echo $test_names) 2>&1 | grep 'found in ' | awk '{print$2}')
              COMPREPLY=($(compgen -W "$(echo ${test_names})" -- "${cur}"))
        ;;
        -t|--tag|-x|--exclude|-p|--prgenv|--system)
            COMPREPLY=()
        ;;
        -c|--checkpath|--module-path|--report-file|--module-mappings|-C|--config-file)
            COMPREPLY=($(compgen -f -- $cur))
    	;;
        --prefix|-o|--output|-s|--stage|--perflogdir)
            COMPREPLY=($(compgen -d -- $cur))
    	;;
    esac
}

complete -o filenames -F _cli_options_completions reframe
