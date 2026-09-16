#!/usr/bin/env bash
set -euo pipefail

# Run in the exact CUDA base with only an empty /out directory writable.
# Keep NVIDIA packages from the immutable base; update signed Ubuntu sources.
test -f /etc/apt/sources.list.d/ubuntu.sources
test -z "$(find /out -mindepth 1 -maxdepth 1 -print -quit)"
mkdir -p /tmp/ubuntu-sources /out/debs/partial /out/repository-metadata /out/package-indexes
cp /etc/apt/sources.list.d/ubuntu.sources /tmp/ubuntu-sources/
options=(
  -o Dir::Etc::sourcelist=-
  -o Dir::Etc::sourceparts=/tmp/ubuntu-sources
  -o Dir::Cache::archives=/out/debs
  -o Acquire::Retries=2
)
apt-get "${options[@]}" update
mapfile -t security_pins < /security-pins.txt
apt-get "${options[@]}" --download-only --yes --no-install-recommends install \
  "${security_pins[@]}" \
  python3 python3-venv python3-dev build-essential libnuma1 libibverbs1 \
  ca-certificates git libgl1
cp /usr/share/keyrings/ubuntu-archive-keyring.gpg /out/package-indexes/
cp /tmp/ubuntu-sources/ubuntu.sources /out/repository-metadata/
cp /var/lib/apt/lists/*InRelease /out/repository-metadata/
sha256sum /out/debs/*.deb > /out/deb-artifacts.sha256
for artifact in /out/debs/*.deb; do
  printf '%s\t' "$(basename "$artifact")"
  dpkg-deb --show --showformat='${Package}\t${Version}\t${Architecture}\n' "$artifact"
done > /out/deb-packages.tsv
dpkg-query -W > /out/base-packages.tsv
chmod -R a+rX /out
