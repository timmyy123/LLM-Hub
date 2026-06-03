#!/bin/bash
# =============================================================================
# Upload patched RunAnywhere iOS SDK XCFrameworks to GitHub Releases
# =============================================================================
#
# USAGE:
#   ./ios/scripts/upload-sdk-release.sh <GITHUB_TOKEN>
#
# WHAT IT DOES:
#   1. Zips the built XCFrameworks from runanywhere-sdks-latest/sdk/runanywhere-commons/dist/
#   2. Creates or updates the GitHub Release on timmyy123/LLM-Hub tagged ios-sdk-v0.19.7-patched-v9
#   3. Uploads both ZIPs + checksums.txt as release assets
#   4. Prints the SHA-256 checksums so you can update Package.swift
#
# REQUIREMENTS:
#   - GITHUB_TOKEN with repo write permission (create release + upload assets)
#   - XCFrameworks already built in dist/ (run build-ios.sh first)
#   - curl, zip, shasum (all standard on macOS)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMMONS_DIR="${REPO_ROOT}/ios/runanywhere-sdks-latest/sdk/runanywhere-commons"
DIST_DIR="${COMMONS_DIR}/dist"

# Config
GITHUB_TOKEN=""
RELEASE_TAG_OVERRIDE=""

# Parse arguments: named flags or positional token
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) RELEASE_TAG_OVERRIDE="$2"; shift 2 ;;
        --notes) shift 2 ;;  # ignore, use default
        --*) shift 2 ;;
        *) GITHUB_TOKEN="$1"; shift ;;
    esac
done

# Keychain fallback if no token provided
if [[ -z "$GITHUB_TOKEN" ]]; then
    GITHUB_TOKEN=$(security find-generic-password -s "GitHub - https://api.github.com" -w 2>/dev/null || \
                   security find-generic-password -s "github-token" -w 2>/dev/null || \
                   security find-internet-password -s "github.com" -w 2>/dev/null || true)
fi

REPO="timmyy123/LLM-Hub"
SDK_VERSION="0.19.7"
RELEASE_TAG="${RELEASE_TAG_OVERRIDE:-ios-sdk-v${SDK_VERSION}-patched-v16}"
RELEASE_TITLE="iOS SDK v${SDK_VERSION} (patched v16)"
RELEASE_NOTES="Patched RunAnywhere SDK v${SDK_VERSION}:
- UPDATE: iOS LlamaCPP build uses official ggml-org/llama.cpp b9493 (clean build, cache cleared, 2026-06-04 UTC).
- REVERT: gpu_layers override logic restored to original unconditional version.
- FIX: Link-complete llama.cpp packaging by bundling libllama-common/libllama-common-base and all mtmd model implementations.
- FIX: Chunked-decode n_cur bug in llamacpp_backend.cpp (line 873). When prompt > n_batch (2048), batch.n_tokens held only the last chunk size instead of total prompt_tokens, causing generated tokens to overwrite existing KV cache entries → 1‑char responses or llama_decode failures.
- FIX: context_size forwarded from model registry → rac_llm_service.cpp → llamacpp_create_service (fixes n_ctx stuck at 1024)
- FIX: Gemma 4 chat prompt format (<|turn>...<turn|>) in ChatScreen multi‑turn builder
- MAX_BATCH_SIZE = 2048, MAX_UBATCH_SIZE = 512
- Gemma 4 VLM prompt and stop‑token fixes
- Built from ios/runanywhere-sdks-latest local source

To use, set ios/runanywhere-sdks-latest/Package.swift useLocalBinaries = false
and update the checksums to match checksums.txt in this release."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[✓]${NC} $1"; }
log_step()  { echo -e "${BLUE}==>${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }
log_warn()  { echo -e "${YELLOW}[!]${NC} $1"; }

if [[ -z "$GITHUB_TOKEN" ]]; then
    echo ""
    echo "Usage: $0 <GITHUB_TOKEN>"
    echo ""
    echo "Create a token at: https://github.com/settings/tokens"
    echo "Required scopes: repo (to create releases and upload assets)"
    exit 1
fi

# Verify XCFrameworks exist
[[ -d "${DIST_DIR}/RACommons.xcframework" ]] || log_error "RACommons.xcframework not found in ${DIST_DIR}. Run build-ios.sh first."
[[ -d "${DIST_DIR}/RABackendLLAMACPP.xcframework" ]] || log_error "RABackendLLAMACPP.xcframework not found in ${DIST_DIR}. Run build-ios.sh first."

# Step 1: Create ZIPs
log_step "Creating ZIP packages..."
cd "${DIST_DIR}"
rm -f "RACommons-v${SDK_VERSION}.zip" "RABackendLLAMACPP-v${SDK_VERSION}.zip" checksums.txt
zip -r "RACommons-v${SDK_VERSION}.zip" RACommons.xcframework -x "*.DS_Store"
zip -r "RABackendLLAMACPP-v${SDK_VERSION}.zip" RABackendLLAMACPP.xcframework -x "*.DS_Store"
shasum -a 256 "RACommons-v${SDK_VERSION}.zip" "RABackendLLAMACPP-v${SDK_VERSION}.zip" > checksums.txt
log_info "ZIPs created:"
ls -lh "RACommons-v${SDK_VERSION}.zip" "RABackendLLAMACPP-v${SDK_VERSION}.zip"
echo ""
echo "=== checksums.txt ==="
cat checksums.txt
echo ""

# Extract individual checksums
COMMONS_CHECKSUM=$(shasum -a 256 "RACommons-v${SDK_VERSION}.zip" | awk '{print $1}')
LLAMACPP_CHECKSUM=$(shasum -a 256 "RABackendLLAMACPP-v${SDK_VERSION}.zip" | awk '{print $1}')

# Step 2: Create GitHub Release
log_step "Creating GitHub Release: ${RELEASE_TAG}..."
API_URL="https://api.github.com/repos/${REPO}/releases"

RELEASE_PAYLOAD=$(cat <<EOF
{
  "tag_name": "${RELEASE_TAG}",
  "name": "${RELEASE_TITLE}",
  "body": $(echo "${RELEASE_NOTES}" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read()))"),
  "prerelease": true,
  "draft": false
}
EOF
)

RELEASE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_URL}" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -H "Content-Type: application/json" \
    -d "${RELEASE_PAYLOAD}")

HTTP_CODE=$(printf '%s\n' "${RELEASE_RESPONSE}" | tail -n 1)
RESPONSE_BODY=$(printf '%s\n' "${RELEASE_RESPONSE}" | sed '$d')

if [[ "$HTTP_CODE" == "201" ]]; then
    log_info "Release created"
elif [[ "$HTTP_CODE" == "422" ]]; then
    # Release already exists — get its upload_url
    log_warn "Release ${RELEASE_TAG} already exists, fetching upload URL..."
    RESPONSE_BODY=$(curl -s "https://api.github.com/repos/${REPO}/releases/tags/${RELEASE_TAG}" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json")
else
    log_error "Failed to create release (HTTP ${HTTP_CODE}):\n${RESPONSE_BODY}"
fi

UPLOAD_URL=$(echo "${RESPONSE_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['upload_url'].replace('{?name,label}',''))")
RELEASE_HTML_URL=$(echo "${RESPONSE_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['html_url'])")

# Step 3: Upload assets
upload_asset() {
    local FILE=$1
    local FILENAME=$(basename "${FILE}")
    local ASSET_ID=""

    ASSET_ID=$(echo "${RESPONSE_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); assets=d.get('assets', []); found=next((str(a['id']) for a in assets if a.get('name') == '${FILENAME}'), ''); print(found)")
    if [[ -n "${ASSET_ID}" ]]; then
        log_step "Deleting existing asset ${FILENAME}..."
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -X DELETE "https://api.github.com/repos/${REPO}/releases/assets/${ASSET_ID}" \
            -H "Authorization: Bearer ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28")
        [[ "$HTTP_CODE" == "204" ]] || log_error "Failed to delete existing asset ${FILENAME} (HTTP ${HTTP_CODE})"
    fi

    log_step "Uploading ${FILENAME}..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${UPLOAD_URL}?name=${FILENAME}" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        -H "Content-Type: application/zip" \
        --data-binary "@${FILE}")
    [[ "$HTTP_CODE" == "201" ]] && log_info "Uploaded ${FILENAME}" || log_warn "Upload returned HTTP ${HTTP_CODE} for ${FILENAME}"
}

upload_text_asset() {
    local FILE=$1
    local FILENAME=$(basename "${FILE}")
    local ASSET_ID=""

    ASSET_ID=$(echo "${RESPONSE_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); assets=d.get('assets', []); found=next((str(a['id']) for a in assets if a.get('name') == '${FILENAME}'), ''); print(found)")
    if [[ -n "${ASSET_ID}" ]]; then
        log_step "Deleting existing asset ${FILENAME}..."
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -X DELETE "https://api.github.com/repos/${REPO}/releases/assets/${ASSET_ID}" \
            -H "Authorization: Bearer ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28")
        [[ "$HTTP_CODE" == "204" ]] || log_error "Failed to delete existing asset ${FILENAME} (HTTP ${HTTP_CODE})"
    fi

    log_step "Uploading ${FILENAME}..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${UPLOAD_URL}?name=${FILENAME}" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        -H "Content-Type: text/plain" \
        --data-binary "@${FILE}")
    [[ "$HTTP_CODE" == "201" ]] && log_info "Uploaded ${FILENAME}" || log_warn "Upload returned HTTP ${HTTP_CODE} for ${FILENAME}"
}

upload_asset "${DIST_DIR}/RACommons-v${SDK_VERSION}.zip"
upload_asset "${DIST_DIR}/RABackendLLAMACPP-v${SDK_VERSION}.zip"
upload_text_asset "${DIST_DIR}/checksums.txt"

# Step 4: Print Package.swift update instructions
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Update ios/runanywhere-sdks-latest/Package.swift:"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  let useLocalBinaries = false"
echo "  let sdkVersion = \"${SDK_VERSION}\""
echo ""
echo "  // In binaryTargets() remote section:"
echo "  .binaryTarget("
echo "      name: \"RACommonsBinary\","
echo "      url: \"https://github.com/${REPO}/releases/download/${RELEASE_TAG}/RACommons-v\(sdkVersion).zip\","
echo "      checksum: \"${COMMONS_CHECKSUM}\""
echo "  ),"
echo "  .binaryTarget("
echo "      name: \"RABackendLlamaCPPBinary\","
echo "      url: \"https://github.com/${REPO}/releases/download/${RELEASE_TAG}/RABackendLLAMACPP-v\(sdkVersion).zip\","
echo "      checksum: \"${LLAMACPP_CHECKSUM}\""
echo "  ),"
echo ""
echo "Release: ${RELEASE_HTML_URL}"
echo ""
log_info "Done! Push the Package.swift changes, then run Xcode File → Packages → Reset Package Caches."
