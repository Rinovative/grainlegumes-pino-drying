#!/usr/bin/env bash

# Shared, domain-labelled prerequisite diagnostics for Generation shell entry points.

generation_prerequisite_missing() {
  local domain="$1"
  local capability="$2"
  local blocked_operation="$3"
  printf '%s prerequisite missing: %s (blocks %s).\n' \
    "${domain}" "${capability}" "${blocked_operation}" >&2
  return 1
}

generation_prerequisite_failed() {
  local domain="$1"
  local capability="$2"
  local blocked_operation="$3"
  printf '%s prerequisite failed: %s (blocks %s).\n' \
    "${domain}" "${capability}" "${blocked_operation}" >&2
  return 1
}

generation_require_command() {
  local domain="$1"
  local command_name="$2"
  local blocked_operation="$3"
  local resolved
  if ! command -v -- "${command_name}" >/dev/null 2>&1; then
    generation_prerequisite_missing \
      "${domain}" "${command_name}" "${blocked_operation}"
    return 1
  fi
  resolved="$(type -P -- "${command_name}" 2>/dev/null || true)"
  if [[ -z "${resolved}" ]]; then
    resolved="shell-$(type -t -- "${command_name}")"
  fi
  printf 'Preflight domain=%s check=command:%s status=pass executable=%s\n' \
    "${domain}" "${command_name}" "${resolved}"
}

generation_report_pass() {
  local domain="$1"
  local check="$2"
  local evidence="${3:-}"
  printf 'Preflight domain=%s check=%s status=pass' "${domain}" "${check}"
  if [[ -n "${evidence}" ]]; then
    printf ' evidence=%s' "${evidence}"
  fi
  printf '\n'
}

generation_run_check() {
  local domain="$1"
  local check="$2"
  local blocked_operation="$3"
  shift 3
  local output
  if ! output="$("$@" 2>&1)"; then
    [[ -z "${output}" ]] || printf '%s\n' "${output}" >&2
    generation_prerequisite_failed \
      "${domain}" "${check}" "${blocked_operation}"
    return 1
  fi
  generation_report_pass "${domain}" "${check}"
  [[ -z "${output}" ]] || printf '%s\n' "${output}"
}
