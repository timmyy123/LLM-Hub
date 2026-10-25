#!/bin/zsh
# Builds LLM Hub for macOS (Release, Apple silicon), signs it with the hardened runtime and puts it in a
# drag-to-Applications DMG. Optionally notarizes and staples the DMG.
#
#   desktop/tools/package/macos/package-macos.sh [--identity NAME] [--notarize PROFILE] [--out DIR] [--work DIR]
#
# --identity  Signing identity (a name or SHA-1 from `security find-identity -v -p codesigning`). Default:
#             $LLMHUB_MAC_IDENTITY, else the first "Developer ID Application" identity in the keychain, else ad-hoc ("-").
#             Only Developer ID builds open on other Macs without a Gatekeeper warning, and only after notarization.
# --notarize  A notarytool keychain profile (see the Packaging section of desktop/README.md). Needs Developer ID.
# --out       Where the DMG goes. Default: desktop/tools/package/out
# --work      Derived data and staging. Default: desktop/tools/package/work/macos (delete it afterwards to free space)
set -euo pipefail

HERE=${0:A:h}
DESKTOP=${HERE:h:h:h}
IDENTITY=${LLMHUB_MAC_IDENTITY:-}
PROFILE=""
OUT=$DESKTOP/tools/package/out
WORK=$DESKTOP/tools/package/work/macos
while (( $# )); do
  case $1 in
    --identity) IDENTITY=$2; shift 2 ;;
    --notarize) PROFILE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

VERSION=$(sed -n 's/^MARKETING_VERSION *= *//p' "$DESKTOP/version.xcconfig")
[[ -n $VERSION ]] || { echo "no MARKETING_VERSION in version.xcconfig" >&2; exit 1; }
if [[ -z $IDENTITY ]]; then
  IDENTITY=$(security find-identity -v -p codesigning | sed -n 's/.*"\(Developer ID Application: .*\)"/\1/p' | head -1)
  [[ -n $IDENTITY ]] || IDENTITY="-"
fi
if [[ -n $PROFILE && $IDENTITY != "Developer ID Application:"* && $IDENTITY != "-" ]]; then
  echo "note: notarization needs a Developer ID Application identity; '$IDENTITY' will be rejected" >&2
fi
echo "LLM Hub $VERSION, signing with: $IDENTITY"

# 1. Release build. Apple silicon only: MLX (Studio music) and Apple Intelligence need it.
cd "$DESKTOP/macos"
xcodegen generate --quiet
mkdir -p "$WORK" "$OUT"
xcodebuild -project LLMHub.xcodeproj -scheme LLMHub -configuration Release -derivedDataPath "$WORK/dd" \
  -destination 'generic/platform=macOS' ARCHS=arm64 ONLY_ACTIVE_ARCH=NO \
  CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM= build -quiet
BUILT=$WORK/dd/Build/Products/Release/LLMHub.app
[[ -d $BUILT ]] || { echo "build produced no LLMHub.app" >&2; exit 1; }

# 2. Stage and sign inside out: nested code first, then the app with its entitlements and the hardened runtime.
STAGE=$WORK/dmg
rm -rf "$STAGE"
mkdir -p "$STAGE"
ditto "$BUILT" "$STAGE/LLMHub.app"
APP=$STAGE/LLMHub.app
ENTITLEMENTS=$DESKTOP/macos/LLMHub/LLMHub.entitlements
SIGN=(codesign --force --sign "$IDENTITY" --options runtime)
if [[ $IDENTITY == "-" ]]; then SIGN+=(--timestamp=none); else SIGN+=(--timestamp); fi

is_macho() { file -b "$1" | grep -q "Mach-O"; }
# Loose Mach-O files (dylibs, helper executables) anywhere except the main executable, deepest first.
find "$APP/Contents" -type f ! -path "$APP/Contents/MacOS/LLMHub" -print0 | while IFS= read -r -d '' f; do
  if is_macho "$f"; then echo "$f"; fi
done | awk '{ print length($0) "\t" $0 }' | sort -rn | cut -f2- | while IFS= read -r f; do
  "${SIGN[@]}" "$f"
done
# Bundles (frameworks, XPC services, plug-ins), deepest first.
find "$APP/Contents" -depth -type d \( -name "*.framework" -o -name "*.xpc" -o -name "*.appex" -o -name "*.bundle" \) -print0 |
  while IFS= read -r -d '' b; do
    [[ -f $b/Contents/Info.plist || -f $b/Info.plist || -f $b/Resources/Info.plist ]] || continue
    # Resource-only bundles (no executable) are sealed by the app's signature instead.
    if [[ -n $(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$b/Contents/Info.plist" 2>/dev/null ||
              /usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$b/Resources/Info.plist" 2>/dev/null ||
              /usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$b/Info.plist" 2>/dev/null) ]]; then
      "${SIGN[@]}" "$b"
    fi
  done
"${SIGN[@]}" --entitlements "$ENTITLEMENTS" "$APP"
codesign --verify --deep --strict --verbose=1 "$APP"
codesign -d --entitlements - --xml "$APP" 2>/dev/null | plutil -convert xml1 -o - - | grep -E "<key>" | sed 's/^[[:space:]]*/  entitlement: /'

# 3. DMG: the app plus a link to /Applications to drag it onto.
ln -s /Applications "$STAGE/Applications"
DMG=$OUT/LLMHub-$VERSION-macos-arm64.dmg
rm -f "$DMG"
# HFS+ (hdiutil sizes APFS images from a folder too small). ULMO (LZMA) is the smallest format macOS 26 opens.
hdiutil create -volname "LLM Hub" -srcfolder "$STAGE" -fs HFS+ -format ULMO -ov "$DMG" -quiet
if [[ $IDENTITY != "-" ]]; then
  codesign --force --sign "$IDENTITY" --timestamp "$DMG"
fi

# 4. Notarize and staple (Developer ID only).
if [[ -n $PROFILE ]]; then
  xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$DMG"
  spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
fi
rm -rf "$STAGE"
echo "$DMG ($(du -h "$DMG" | cut -f1))"
