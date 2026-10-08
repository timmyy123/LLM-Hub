#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 1
fi

OUTPUT_DIR="$1"
GIT_ROOT="$(git rev-parse --show-toplevel)"
WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mgk-docc.XXXXXX")"
WORKTREE_PATH="${WORK_ROOT}/repo"
ARCHIVE_PATH="${WORK_ROOT}/docs"

cleanup() {
  git -C "${GIT_ROOT}" worktree remove --force "${WORKTREE_PATH}" >/dev/null 2>&1 || true
  rm -rf "${WORK_ROOT}"
}
trap cleanup EXIT

git -C "${GIT_ROOT}" worktree add --detach "${WORKTREE_PATH}" HEAD >/dev/null

cd "${WORKTREE_PATH}"

if ! git -C "${GIT_ROOT}" diff --quiet --binary HEAD --; then
  git -C "${GIT_ROOT}" diff --binary HEAD -- | git apply --allow-empty -p1
fi

while IFS= read -r -d '' path; do
  mkdir -p "${WORKTREE_PATH}/$(dirname "${path}")"
  cp -R "${GIT_ROOT}/${path}" "${WORKTREE_PATH}/${path}"
done < <(git -C "${GIT_ROOT}" ls-files --others --exclude-standard -z)

python3 - <<'PY'
from pathlib import Path

package = Path("Package.swift")
text = package.read_text()
text = text.replace('name: "DrawThings"', 'name: "MediaGenerationKit"', 1)
text = text.replace(
    '.library(name: "_MediaGenerationKit", targets: ["_MediaGenerationKit"])',
    '.library(name: "MediaGenerationKit", targets: ["MediaGenerationKit"])',
    1,
)
text = text.replace('name: "_MediaGenerationKit"', 'name: "MediaGenerationKit"', 1)
package.write_text(text)

module_page = Path("Libraries/MediaGenerationKit/Sources/MediaGenerationKit.docc/MediaGenerationKit.md")
module_page.write_text(
    module_page.read_text().replace('# ``_MediaGenerationKit``', '# ``MediaGenerationKit``', 1)
)
PY

rm -rf "${ARCHIVE_PATH}" "${OUTPUT_DIR}"

swift package generate-documentation \
  --target MediaGenerationKit \
  --output-path "${ARCHIVE_PATH}" \
  --disable-indexing \
  --hosting-base-path /media-generation-kit \
  --transform-for-static-hosting

mkdir -p "${OUTPUT_DIR}"
cp -R "${ARCHIVE_PATH}/." "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/index.html" <<'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0;url=./documentation/mediagenerationkit">
    <title>MediaGenerationKit Documentation</title>
  </head>
  <body>
    <p>Redirecting to <a href="./documentation/mediagenerationkit">MediaGenerationKit documentation</a>...</p>
  </body>
</html>
EOF

touch "${OUTPUT_DIR}/.nojekyll"
