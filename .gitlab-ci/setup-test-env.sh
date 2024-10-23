#!/usr/bin/env bash
# shellcheck disable=SC2048
# shellcheck disable=SC2086 # we want word splitting
# shellcheck disable=SC2155 # mktemp usually not failing

shopt -s expand_aliases

function _x_off {
    if [[ "$-" == *"x"* ]]; then
      state_x=1
      set +x
    else
      state_x=0
    fi
}

alias x_off='{ _x_off; } >/dev/null 2>/dev/null'

# TODO: implement x_on !

export JOB_START_S=$(date -u +"%s" -d "${CI_JOB_STARTED_AT:?}")

function get_current_minsec {
    DATE_S=$(date -u +"%s")
    CURR_TIME=$((DATE_S-JOB_START_S))
    printf "%02d:%02d" $((CURR_TIME/60)) $((CURR_TIME%60))
}

function error {
    x_off 2>/dev/null
    RED="\e[0;31m"
    ENDCOLOR="\e[0m"
    # we force the following to be not in a section
    _section_end $CURRENT_SECTION

    CURR_MINSEC=$(get_current_minsec)
    echo -e "\n${RED}[${CURR_MINSEC}] ERROR: $*${ENDCOLOR}\n"
    [ "$state_x" -eq 0 ] || set -x
}

function trap_err {
    error ${CURRENT_SECTION:-'unknown-section'}: ret code: $*
}

function _build_section_start {
    local section_params=$1
    shift
    local section_name=$1
    CURRENT_SECTION=$section_name
    shift
    CYAN="\e[0;36m"
    ENDCOLOR="\e[0m"

    CURR_MINSEC=$(get_current_minsec)
    echo -e "\n\e[0Ksection_start:$(date +%s):$section_name$section_params\r\e[0K${CYAN}[${CURR_MINSEC}] $*${ENDCOLOR}\n"
}
alias build_section_start="x_off; _build_section_start"

function _section_start {
    _build_section_start "[collapsed=true]" $*
    [ "$state_x" -eq 0 ] || set -x
}
alias section_start="x_off; _section_start"

function _uncollapsed_section_start {
    _build_section_start "" $*
    [ "$state_x" -eq 0 ] || set -x
}
alias uncollapsed_section_start="x_off; _uncollapsed_section_start"

function _build_section_end {
    echo -e "\e[0Ksection_end:$(date +%s):$1\r\e[0K"
    CURRENT_SECTION=""
}
alias build_section_end="x_off; _build_section_end"

function _section_end {
    _build_section_end $*
    [ "$state_x" -eq 0 ] || set -x
}
alias section_end="x_off; _section_end"

function _section_switch {
    if [ -n "$CURRENT_SECTION" ]
    then
	_build_section_end $CURRENT_SECTION
    fi
    _build_section_start "[collapsed=true]" $*
    [ "$state_x" -eq 0 ] || set -x
}
alias section_switch="x_off; _section_switch"

function _uncollapsed_section_switch {
    if [ -n "$CURRENT_SECTION" ]
    then
	_build_section_end $CURRENT_SECTION
    fi
    _build_section_start "" $*
    [ "$state_x" -eq 0 ] || set -x
}
alias uncollapsed_section_switch="x_off; _uncollapsed_section_switch"

export -f _x_off
export -f get_current_minsec
export -f error
export -f trap_err
export -f _build_section_start
export -f _section_start
export -f _build_section_end
export -f _section_end
export -f _section_switch
export -f _uncollapsed_section_switch

# Freedesktop requirement (needed for Wayland)
[ -n "${XDG_RUNTIME_DIR:-}" ] || export XDG_RUNTIME_DIR="$(mktemp -p "$PWD" -d xdg-runtime-XXXXXX)"

if [ -z "${RESULTS_DIR:-}" ]; then
	export RESULTS_DIR="$(pwd)/results"
	if [ -e "${RESULTS_DIR}" ]; then
		rm -rf "${RESULTS_DIR}"
	fi
	mkdir -p "${RESULTS_DIR}"
fi

set -E
trap 'trap_err $?' ERR
