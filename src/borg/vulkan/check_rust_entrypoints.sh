#!/bin/sh
# Every entry point defined in rust/ must be in the driver: the dispatch table's references are
# weak, and a missing one silently falls back to the runtime's vk_common_* or to NULL.
so="$1"; rs="$2"
syms=$(nm "$so" | awk '$2 ~ /^[tT]$/ {print $3}')
missing=""
for f in $(grep -ho 'fn borgvk_[A-Z][A-Za-z0-9]*' "$rs"/*.rs | sed 's/fn //' | sort -u); do
  echo "$syms" | grep -qx "$f" || missing="$missing $f"
done
[ -z "$missing" ] || { echo "not in $so:$missing"; exit 1; }
