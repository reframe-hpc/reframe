#/usr/bin/env bash

_bin=/home/sarafael/repos/reframe/bin/reframe

_cli_options=$(${_bin} --help | grep -o -e ' --[a-z\-]*' -e ' -[a-zA-Z]' --)

_cli_options_completions()
{
  local cur prev
   
  cur=${COMP_WORDS[COMP_CWORD]}
  prev=${COMP_WORDS[COMP_CWORD-1]}

  COMPREPLY=($(compgen -W "${_cli_options}" -- "${cur}"))

  case ${prev} in
	  -n|--name)
		  # exclude the -n from the command line to get the test names
          local COMP_WORDS_NO_FLAGS=()
          local index=0
		  while [[ "$index" -lt "$((COMP_CWORD-1))" ]]
          do
              COMP_WORDS_NO_FLAGS+=("${COMP_WORDS[$index]}")
              let index++
          done

		  local subfunction=$(echo "${COMP_WORDS_NO_FLAGS[*]} -l")
		  local names=$($(echo ${subfunction} | tr % ' ') | grep 'found in ' | awk '{print$2}')

		  COMPREPLY=($(compgen -W "$(echo ${names})" -- "${cur}"))
	  ;;
      -c|--checkpath|-C|--config-file)
          COMPREPLY=( $( compgen -f -- $cur ) )
          # COMPREPLY=( $( compgen -o plusdirs -f -X '!*.py' -- $cur ) )
	  ;;
  esac
}

complete -o filenames -F _cli_options_completions reframe
# complete -o filenames -o dirnames -F _cli_options_completions reframe




# _cli_options=$(/home/sarafael/repos/reframe/bin/reframe --help | grep -o -e ' --[a-z\-]*' -e ' -[a-zA-Z]' --)
# 
# _cli_options_completions()
# {
#   COMPREPLY=($(compgen -W "${_cli_options}" -- "${COMP_WORDS[COMP_CWORD]}"))
# }
# 
# complete -F _cli_options_completions reframe
