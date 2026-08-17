#!/usr/bin/env bash
set -Eeuo pipefail

HOST_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
HOST_STORAGE_ROOT="${STORAGE_ROOT:-${HOST_REPO_ROOT}/../storage}"
IMAGE_NAME="grainlegumes-pino-drying"

fail() {
  local status="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "${status}"
}

command -v docker >/dev/null 2>&1 || fail 1 "Local project Python requires Docker on the host."
docker info >/dev/null 2>&1 || fail 1 "Local project Python requires the Docker daemon."
docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 ||
  fail 1 "Local project Python requires image '${IMAGE_NAME}'. Build it with ./scripts/docker_build.sh."

mkdir -p "${HOST_STORAGE_ROOT}"
STORAGE_DIR="$(cd "${HOST_STORAGE_ROOT}" && pwd -P)"

translate_argument() {
  local argument="$1"
  if [[ "${argument}" == "${HOST_REPO_ROOT}" ]]; then
    printf '/workspace/repo'
  elif [[ "${argument}" == "${HOST_REPO_ROOT}/"* ]]; then
    printf '/workspace/repo/%s' "${argument#"${HOST_REPO_ROOT}/"}"
  elif [[ "${argument}" == "${STORAGE_DIR}" ]]; then
    printf '/workspace/storage'
  elif [[ "${argument}" == "${STORAGE_DIR}/"* ]]; then
    printf '/workspace/storage/%s' "${argument#"${STORAGE_DIR}/"}"
  else
    printf '%s' "${argument}"
  fi
}

PYTHON_ARGUMENTS=()
for argument in "$@"; do
  PYTHON_ARGUMENTS+=("$(translate_argument "${argument}")")
done

COMMIT_ENVIRONMENT=()
if [[ -n "${GENERATION_GIT_COMMIT:-}" ]]; then
  COMMIT_ENVIRONMENT+=(--env "GENERATION_GIT_COMMIT=${GENERATION_GIT_COMMIT}")
fi

exec docker run --rm -i \
  --network none \
  --user "$(id -u):$(id -g)" \
  --workdir /workspace/repo \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  -e HOME=/tmp \
  -e PROJECT_ROOT=/workspace/repo \
  -e STORAGE_ROOT=/workspace/storage \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONNOUSERSITE=1 \
  "${COMMIT_ENVIRONMENT[@]}" \
  --mount "type=bind,source=${HOST_REPO_ROOT},target=/workspace/repo,readonly" \
  --mount "type=bind,source=${STORAGE_DIR},target=/workspace/storage" \
  "${IMAGE_NAME}" \
  python "${PYTHON_ARGUMENTS[@]}"
