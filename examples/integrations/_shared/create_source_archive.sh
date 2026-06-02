#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  examples/integrations/_shared/create_source_archive.sh [--output PATH] [--committed|--include-worktree]

Create a source-only tarball for outreach-style integration validation.

Modes:
  --committed          Archive HEAD with git archive. This matches GitHub source
                       archives and excludes uncommitted worktree changes.
  --include-worktree   Archive tracked, modified, staged, and untracked
                       non-ignored files from the current checkout.

Default:
  If the checkout is clean, use --committed. If it is dirty, use
  --include-worktree so local pre-PR validation can include pending changes.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

output="${TMPDIR:-/tmp}/invarlock-current-source.tgz"
mode="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output="${2:-}"
      shift 2
      ;;
    --committed)
      mode="committed"
      shift
      ;;
    --include-worktree)
      mode="worktree"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$output" ]]; then
  echo "--output requires a path." >&2
  exit 2
fi

mkdir -p "$(dirname -- "$output")"

if [[ "$mode" == "auto" ]]; then
  if git -C "$REPO_ROOT" diff --quiet --ignore-submodules -- \
    && git -C "$REPO_ROOT" diff --cached --quiet --ignore-submodules -- \
    && [[ -z "$(git -C "$REPO_ROOT" ls-files --others --exclude-standard)" ]]; then
    mode="committed"
  else
    mode="worktree"
  fi
fi

case "$mode" in
  committed)
    git -C "$REPO_ROOT" archive --format=tar.gz --output "$output" HEAD
    ;;
  worktree)
    tar_bin="${TAR:-}"
    if [[ -z "$tar_bin" ]]; then
      if command -v tar >/dev/null 2>&1; then
        tar_bin="$(command -v tar)"
      elif [[ -x /usr/bin/tar ]]; then
        tar_bin="/usr/bin/tar"
      else
        echo "tar is required for --include-worktree archives." >&2
        exit 1
      fi
    fi

    file_list="$(mktemp "${TMPDIR:-/tmp}/invarlock-source-files.XXXXXX")"
    trap 'rm -f "$file_list"' EXIT

    git -C "$REPO_ROOT" ls-files -z --cached --modified --others --exclude-standard \
      | while IFS= read -r -d '' path; do
          if [[ -e "$REPO_ROOT/$path" ]]; then
            printf '%s\0' "$path"
          fi
        done > "$file_list"

    if [[ ! -s "$file_list" ]]; then
      echo "No source files found to archive." >&2
      exit 1
    fi

    tar_args=()
    if "$tar_bin" --no-xattrs -cf /dev/null -T /dev/null >/dev/null 2>&1; then
      tar_args+=(--no-xattrs)
    fi

    (
      cd "$REPO_ROOT"
      COPYFILE_DISABLE=1 "$tar_bin" "${tar_args[@]}" -czf "$output" --null -T "$file_list"
    )
    ;;
  *)
    echo "Unknown archive mode: $mode" >&2
    exit 2
    ;;
esac

printf 'Wrote source archive: %s\n' "$output"
