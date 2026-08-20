#!/usr/bin/env bash

# Shared one-shot and bounded-retry SSH transport for Generation host workflows.

_GENERATION_SSH_DEFAULT_RETRY_DELAYS="5,15,30,60"
_GENERATION_SSH_MAX_CLASSIFIER_BYTES=32768
_GENERATION_SSH_MAX_REASON_CHARACTERS=240

remote_bash_once() {
  local host="$1"
  shift
  local command="bash -l -s --"
  local argument quoted
  for argument in "$@"; do
    printf -v quoted '%q' "${argument}"
    command+=" ${quoted}"
  done
  ssh -o BatchMode=yes -- "${host}" "${command}"
}

_generation_ssh_retry_temp_parent() {
  printf '%s\n' /tmp
}

_generation_ssh_retry_sleep() {
  sleep "$1"
}

_generation_ssh_retry_reason() {
  local status="$1" stderr_path="$2"
  (( status == 255 )) || return 1
  local sample
  sample="$(head -c "${_GENERATION_SSH_MAX_CLASSIFIER_BYTES}" -- "${stderr_path}")" || return 1
  case "${sample}" in
    *"REMOTE HOST IDENTIFICATION HAS CHANGED"* | \
      *"Host key verification failed"* | \
      *"Bad owner or permissions"* | \
      *"Bad configuration option"* | \
      *"Identity file"*"not accessible"* | \
      *"Traceback (most recent call last)"*)
      return 1
      ;;
  esac
  local line reason
  while IFS= read -r line || [[ -n "${line}" ]]; do
    case "${line}" in
      *"Permission denied ("* | \
        *"Connection timed out"* | \
        *"Connection reset by peer"* | \
        *"Connection closed by "* | \
        *"Connection refused"* | \
        *"No route to host"* | \
        *"Broken pipe"* | \
        *"kex_exchange_identification"* | \
        *"Temporary failure in name resolution"* | \
        *"Could not resolve hostname"*)
        reason="$(printf '%s' "${line}" | LC_ALL=C tr -cd '[:print:]\t' | head -c "${_GENERATION_SSH_MAX_REASON_CHARACTERS}")"
        reason="${reason//$'\t'/ }"
        reason="${reason//\"/\'}"
        [[ -n "${reason}" ]] || return 1
        printf '%s\n' "${reason}"
        return 0
        ;;
    esac
  done <<< "${sample}"
  return 1
}

_remote_bash_retryable_with_delays() (
  local operation="$1" delay_text="$2" host="$3"
  shift 3
  if [[ -z "${operation}" || "${operation}" == *$'\n'* || \
    "${operation}" == *$'\r'* || "${operation}" == *$'\t'* ]]; then
    printf 'Invalid Generation SSH retry operation label.\n' >&2
    return 2
  fi
  if [[ ! "${host}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid Generation SSH retry host.\n' >&2
    return 2
  fi
  if [[ ! "${delay_text}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'Invalid Generation SSH retry delay policy.\n' >&2
    return 2
  fi
  local -a retry_delays=()
  IFS=',' read -r -a retry_delays <<< "${delay_text}"
  if (( ${#retry_delays[@]} < 1 || ${#retry_delays[@]} > 15 )); then
    printf 'Generation SSH retry policy must contain between 1 and 15 delays.\n' >&2
    return 2
  fi
  local delay
  for delay in "${retry_delays[@]}"; do
    if [[ ! "${delay}" =~ ^[0-9]+$ ]] || (( delay > 3600 )); then
      printf 'Generation SSH retry delays must be bounded non-negative integer seconds.\n' >&2
      return 2
    fi
  done

  local temporary_parent temporary_directory="" script_path="" stdout_path="" stderr_path=""
  temporary_parent="$(_generation_ssh_retry_temp_parent)" || return $?
  if [[ "${temporary_parent}" != /* || ! -d "${temporary_parent}" || -L "${temporary_parent}" ]]; then
    printf 'Generation SSH retry temporary parent is unsafe.\n' >&2
    return 2
  fi
  _generation_ssh_retry_cleanup() {
    if [[ -n "${script_path}" ]]; then
      rm -f -- "${script_path}" "${stdout_path}" "${stderr_path}"
    fi
    if [[ -n "${temporary_directory}" ]]; then
      rmdir -- "${temporary_directory}" 2>/dev/null || true
    fi
  }
  trap 'status=$?; trap - EXIT; _generation_ssh_retry_cleanup; exit "${status}"' EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM

  temporary_directory="$(mktemp -d -- "${temporary_parent%/}/generation-ssh-retry.XXXXXX")" || return $?
  script_path="${temporary_directory}/remote-script"
  stdout_path="${temporary_directory}/stdout"
  stderr_path="${temporary_directory}/stderr"
  chmod 700 -- "${temporary_directory}"
  umask 077
  : > "${script_path}"
  : > "${stdout_path}"
  : > "${stderr_path}"
  chmod 600 -- "${script_path}" "${stdout_path}" "${stderr_path}"
  cat > "${script_path}"

  local attempt=1 attempts=$(( ${#retry_delays[@]} + 1 )) status reason retry_delay
  while (( attempt <= attempts )); do
    : > "${stdout_path}"
    : > "${stderr_path}"
    if remote_bash_once "${host}" "$@" < "${script_path}" > "${stdout_path}" 2> "${stderr_path}"; then
      cat -- "${stdout_path}"
      cat -- "${stderr_path}" >&2
      if (( attempt > 1 )); then
        printf 'SSH connection recovered:\n  operation=%s\n  host=%s\n  attempts=%s\n' \
          "${operation}" "${host}" "${attempt}" >&2
      fi
      return 0
    else
      status=$?
    fi
    if ! reason="$(_generation_ssh_retry_reason "${status}" "${stderr_path}")"; then
      cat -- "${stderr_path}" >&2 || true
      return "${status}"
    fi
    if (( attempt == attempts )); then
      cat -- "${stderr_path}" >&2 || true
      printf 'Persistent SSH transport/authentication failure after %s attempts during %s.\n' \
        "${attempts}" "${operation}" >&2
      printf 'No CPU-side campaign or Slurm cancellation was requested by the transport retry.\n' >&2
      return "${status}"
    fi
    retry_delay="${retry_delays[attempt - 1]}"
    printf 'WARNING: transient SSH failure during %s\n' "${operation}" >&2
    printf '         host=%s attempt=%s/%s retry_in=%ss\n' \
      "${host}" "${attempt}" "${attempts}" "${retry_delay}" >&2
    printf '         reason="%s"\n' "${reason}" >&2
    if _generation_ssh_retry_sleep "${retry_delay}"; then
      :
    else
      status=$?
      return "${status}"
    fi
    attempt=$(( attempt + 1 ))
  done
  return 255
)

remote_bash_retryable() {
  local operation="$1"
  shift
  _remote_bash_retryable_with_delays \
    "${operation}" "${_GENERATION_SSH_DEFAULT_RETRY_DELAYS}" "$@"
}
