#!/usr/bin/env bash

# Shared, domain-labelled prerequisite diagnostics for Generation shell entry points.
# GENERATION_PREFLIGHT_QUIET=1 silences passing checks only; a missing or failing prerequisite is always reported.

generation_preflight_quiet() {
  [[ "${GENERATION_PREFLIGHT_QUIET:-}" == 1 ]]
}

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
  generation_preflight_quiet && return 0
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
  generation_preflight_quiet && return 0
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
  generation_preflight_quiet || [[ -z "${output}" ]] || printf '%s\n' "${output}"
}

generation_comsol_version() (
  local executable="$1" scratch_parent version_scratch output
  scratch_parent="${TMPDIR:-/tmp}"
  if [[ "${scratch_parent}" != /* || ! -d "${scratch_parent}" || ! -w "${scratch_parent}" ]]; then
    printf 'Native COMSOL version scratch is unavailable: %s\n' "${scratch_parent}" >&2
    return 1
  fi
  version_scratch="$(mktemp -d "${scratch_parent%/}/generation-comsol-version.XXXXXXXX")" || return 1
  trap 'rm -rf -- "${version_scratch}"' EXIT
  if ! output="$("${executable}" -configuration "${version_scratch}" -version 2>&1)"; then
    printf 'Native COMSOL version query failed: %s\n' "${output}" >&2
    return 1
  fi
  if [[ ! "${output}" =~ ^COMSOL\ Multiphysics\ 6\.4(\.[0-9]+)+$ ]]; then
    printf 'Native COMSOL version query returned invalid v6.4 evidence: %s\n' "${output}" >&2
    return 1
  fi
  printf '%s\n' "${output}"
)

generation_validate_native_venv() {
  local domain="$1"
  local venv="$2"
  local blocked_operation="$3"
  if [[ ! -f "${venv}/bin/python" || ! -x "${venv}/bin/python" ]]; then
    generation_prerequisite_missing \
      "${domain}" "executable Generation venv launcher: ${venv}/bin/python" \
      "${blocked_operation}"
    return 1
  fi
  generation_run_check \
    "${domain}" "Generation-venv-runtime" "${blocked_operation}" \
    "${venv}/bin/python" -c \
    'import sys
from src.generation.runtime import preflight
try:
    preflight.validate_generation_venv(sys.argv[1], domain=sys.argv[2])
except Exception as error:
    print(error, file=sys.stderr)
    raise SystemExit(1) from None' \
    "${venv}" "${domain}"
}

# A worker runs from a frozen source tree below RUNTIME_ROOT/sources with explicit roots; no sibling default applies.
generation_validate_worker_roots() {
  local repository="$1"
  local tree_name="${repository##*/}"
  local variable_name value physical
  for variable_name in STORAGE_ROOT RUNTIME_ROOT; do
    value="${!variable_name:-}"
    physical="$(realpath -e -- "${value:-/nonexistent}" 2>/dev/null || true)"
    if [[ "${value}" != /* || ! -d "${value}" || "${value}" != "${physical}" ]]; then
      generation_prerequisite_failed \
        "CPU compute-node" "explicit physical ${variable_name} directory: ${value:-unset}" \
        "repository-owned worker dependencies"
      return 1
    fi
  done
  if [[ ! "${tree_name}" =~ ^[0-9a-f]{12}-[0-9a-f]{12}$ \
    || "${SOURCE_TREE_ID:-}" != "${tree_name}" \
    || "${repository}" != "${RUNTIME_ROOT}/sources/${tree_name}" \
    || "${PROJECT_ROOT:-}" != "${repository}" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "source tree ${RUNTIME_ROOT}/sources/${SOURCE_TREE_ID:-unset} as REPOSITORY and PROJECT_ROOT: ${repository}" \
      "repository-owned worker dependencies"
    return 1
  fi
  if [[ "${GENERATION_NATIVE_VENV:-}" != "${RUNTIME_ROOT}/venvs/native" \
    || ! -x "${RUNTIME_ROOT}/venvs/native/bin/python" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "native Generation venv RUNTIME_ROOT/venvs/native with an executable python" \
      "repository-owned worker dependencies"
    return 1
  fi
}

generation_validate_worker_repository() {
  local repository="$1"
  local expected_commit="$2"
  local runtime_script="$3"
  local helper_path="${repository}/scripts/generation_prerequisites.sh"
  local repository_physical scripts_physical verification tree_commit tree_sha

  if [[ "${repository}" != /* || "${repository}" == / \
    || "${repository}" == *$'\n'* || "${repository}" == *$'\r'* || "${repository}" == *$'\t'* ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "explicit source tree: ${repository}" \
      "repository-owned worker dependencies; Slurm script: ${runtime_script}"
    return 1
  fi
  if [[ ! "${expected_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "exact launch commit" "repository-owned worker dependencies"
    return 1
  fi
  if [[ ! -d "${repository}" || -L "${repository}" || ! -r "${repository}" || ! -x "${repository}" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "safe source tree: ${repository}" \
      "repository-owned worker dependencies; Slurm script: ${runtime_script}"
    return 1
  fi
  if ! repository_physical="$(cd "${repository}" && pwd -P)"; then
    generation_prerequisite_failed \
      "CPU compute-node" "source tree: ${repository}" \
      "repository-owned worker dependencies"
    return 1
  fi
  if [[ "${repository_physical}" != "${repository}" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "non-canonical source tree: ${repository}" \
      "repository-owned worker dependencies"
    return 1
  fi
  if [[ ! -d "${repository}/scripts" || -L "${repository}/scripts" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "safe tree scripts directory: ${repository}/scripts" \
      "repository-owned worker dependencies"
    return 1
  fi
  if ! scripts_physical="$(cd "${repository}/scripts" && pwd -P)"; then
    generation_prerequisite_failed \
      "CPU compute-node" "source tree scripts directory" \
      "repository-owned worker dependencies"
    return 1
  fi
  if [[ "${scripts_physical}" != "${repository}/scripts" \
    || ! -f "${helper_path}" || -L "${helper_path}" || ! -r "${helper_path}" \
    || "${BASH_SOURCE[0]}" != "${helper_path}" ]]; then
    printf 'CPU compute-node prerequisite failed: tree helper missing or unreadable: scripts/generation_prerequisites.sh (source tree: %s; Slurm script: %s).\n' \
      "${repository}" "${runtime_script}" >&2
    return 1
  fi
  if [[ ! "${GENERATION_SOURCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "GENERATION_SOURCE_SHA256 launch fingerprint" \
      "repository-owned worker dependencies"
    return 1
  fi
  if [[ ! -f "${repository}/scripts/source_fingerprint.py" \
    || -L "${repository}/scripts/source_fingerprint.py" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "source tree verification helper" \
      "repository-owned worker dependencies"
    return 1
  fi
  generation_validate_worker_roots "${repository}" || return 1
  # The tree has no Git metadata: its bytes, paths and modes are checked against the durable snapshot named like it.
  if ! verification="$(python3 "${repository}/scripts/source_fingerprint.py" \
    --verify-tree "${repository}" "${STORAGE_ROOT}/source_snapshots")"; then
    generation_prerequisite_failed \
      "CPU compute-node" "source tree verified against its snapshot" \
      "repository-owned worker dependencies"
    return 1
  fi
  if ! read -r tree_commit tree_sha < <(
    python3 -c 'import json, sys
verified = json.load(sys.stdin)
print(verified["git_commit"], verified["source_fingerprint_sha256"])' <<<"${verification}"
  ); then
    generation_prerequisite_failed \
      "CPU compute-node" "readable source tree verification result" \
      "repository-owned worker dependencies"
    return 1
  fi
  if [[ "${tree_commit}" != "${expected_commit}" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "source commit ${expected_commit}" \
      "repository-owned worker dependencies; tree snapshot records ${tree_commit}"
    return 1
  fi
  if [[ "${tree_sha}" != "${GENERATION_SOURCE_SHA256}" ]]; then
    generation_prerequisite_failed \
      "CPU compute-node" "source tree matching the submission fingerprint" \
      "repository-owned worker dependencies"
    return 1
  fi
  generation_report_pass \
    "CPU compute-node" "source-tree" "${repository}@${expected_commit}:${tree_sha}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if (( $# != 4 )) || [[ "$1" != validate-worker-repository ]]; then
    printf 'Usage: %s validate-worker-repository TREE COMMIT RUNTIME_SCRIPT\n' "$0" >&2
    exit 2
  fi
  generation_validate_worker_repository "$2" "$3" "$4"
fi
