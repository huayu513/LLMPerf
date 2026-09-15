#!/usr/bin/env bash

profile_dir_path() {
  printf '%s\n' "${S1_PROFILE_DIR:-${SCRIPT_DIR}/profiles}"
}

profile_exists() {
  local profile="$1"
  [[ "$profile" =~ ^[A-Za-z0-9_-]+$ ]] || return 1
  [[ -f "$(profile_dir_path)/${profile}.sh" ]]
}

profile_names() {
  local path name
  shopt -s nullglob
  for path in "$(profile_dir_path)"/*.sh; do
    name="${path##*/}"
    name="${name%.sh}"
    [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] && printf '%s\n' "$name"
  done | sort
  shopt -u nullglob
}
