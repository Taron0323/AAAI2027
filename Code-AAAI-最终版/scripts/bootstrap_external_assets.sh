#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-submodule}"
manifest="third_party/assets.yaml"
export PYTHONPATH="${PYTHONPATH:-src}"
failures=0

fetch_archive() {
  local name="$1"
  local archive_url="$2"
  local local_path="$3"
  local source_url="$4"
  local source_commit="${5:-}"

  local tmp_archive
  local tmp_dir
  tmp_archive="$(mktemp -t "foreact-${name}.XXXXXX.tar.gz")"
  tmp_dir="$(mktemp -d -t "foreact-${name}.XXXXXX")"

  if curl --http1.1 -L --fail --retry 8 --retry-delay 5 --connect-timeout 30 \
      --speed-time 180 --speed-limit 512 "$archive_url" -o "$tmp_archive"; then
    tar -xzf "$tmp_archive" -C "$tmp_dir"
    local extracted
    extracted="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -1)"
    if [[ -z "$extracted" ]]; then
      echo "archive had no top-level directory: $name" >&2
      rm -f "$tmp_archive"
      rm -rf "$tmp_dir"
      return 1
    fi
    if [[ -e "$local_path" ]]; then
      if [[ -z "$(find "$local_path" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
        rmdir "$local_path"
      else
        echo "target exists; leaving it in place: $local_path"
        rm -f "$tmp_archive"
        rm -rf "$tmp_dir"
        return 0
      fi
    fi
    mkdir -p "$(dirname "$local_path")"
    mv "$extracted" "$local_path"
    {
      printf 'repo: %s\n' "$source_url"
      if [[ -n "$source_commit" ]]; then
        printf 'commit: %s\n' "$source_commit"
      fi
      printf 'source: archive\n'
      date -u '+fetched_at: %Y-%m-%dT%H:%M:%SZ'
    } > "$local_path/.foreact_archive_source"
    echo "downloaded archive: $name"
  else
    echo "archive download failed: $name ($archive_url)" >&2
    rm -f "$tmp_archive"
    rm -rf "$tmp_dir"
    return 1
  fi
  rm -f "$tmp_archive"
  rm -rf "$tmp_dir"
}

fetch_zip_archive() {
  local name="$1"
  local archive_url="$2"
  local local_path="$3"

  local tmp_zip
  local tmp_dir
  tmp_zip="$(mktemp -t "foreact-${name}.XXXXXX.zip")"
  tmp_dir="$(mktemp -d -t "foreact-${name}.XXXXXX")"

  if curl -L --fail --retry 2 --connect-timeout 20 --max-time 180 "$archive_url" -o "$tmp_zip"; then
    unzip -q "$tmp_zip" -d "$tmp_dir"
    local extracted
    extracted="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -1)"
    if [[ -n "$extracted" && ! -e "$local_path" ]]; then
      mv "$extracted" "$local_path"
      echo "downloaded archive: $name"
    else
      echo "archive had no top-level directory: $name" >&2
      rm -f "$tmp_zip"
      rm -rf "$tmp_dir"
      return 1
    fi
  else
    echo "archive download failed: $name ($archive_url)" >&2
    rm -f "$tmp_zip"
    rm -rf "$tmp_dir"
    return 1
  fi
  rm -f "$tmp_zip"
  rm -rf "$tmp_dir"
}

if [[ ! -f "$manifest" ]]; then
  echo "missing $manifest" >&2
  exit 2
fi

while IFS=$'\t' read -r group name url archive_url local_path source_commit; do
  if [[ -e "$local_path/README.md" || -e "$local_path/.git" || -e "$local_path/.foreact_archive_source" ]]; then
    echo "present: $name at $local_path"
    continue
  fi
  if [[ -e "$local_path" ]]; then
    echo "target exists but does not look complete: $local_path" >&2
    echo "inspect it manually, then remove or repair it before retrying $name" >&2
    failures=$((failures + 1))
    continue
  fi
  mkdir -p "$(dirname "$local_path")"
  echo "fetching: $name -> $local_path"
  if [[ "$mode" == "nested" ]]; then
    if ! fetch_archive "$name" "$archive_url" "$local_path" "$url" "$source_commit"; then
      failures=$((failures + 1))
    fi
  elif [[ "$mode" == "archive" ]]; then
    if ! fetch_zip_archive "$name" "$archive_url" "$local_path"; then
      failures=$((failures + 1))
    fi
  else
    if git -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 submodule add --depth 1 "$url" "$local_path"; then
      echo "added submodule: $name"
    else
      echo "submodule add failed: $name ($url)" >&2
      echo "retry with: bash scripts/bootstrap_external_assets.sh archive" >&2
      failures=$((failures + 1))
    fi
  fi
done < <(FOREACT_ASSET_MODE="$mode" PYTHONPATH=src python3 - <<'PY'
import os

from foreact.io import load_yaml

manifest = load_yaml("third_party/assets.yaml")
mode = os.environ.get("FOREACT_ASSET_MODE", "submodule")
groups = ("nested_archive_code",) if mode == "nested" else ("benchmark_code", "baseline_code", "auxiliary_code")
for group in groups:
    for name, spec in manifest.get(group, {}).items():
        print("\t".join([
            group,
            name,
            spec["url"],
            spec["archive_url"],
            spec["local_path"],
            spec.get("commit", ""),
        ]))
PY
)

python3 -m foreact.cli asset-status --manifest "$manifest" --out outputs/smoke/asset_status.json
echo "wrote outputs/smoke/asset_status.json"
if [[ "$failures" -gt 0 ]]; then
  echo "external asset fetch completed with $failures failure(s)" >&2
  exit 1
fi
