#!/bin/bash
# Builds LLM Hub for Linux x64 (self-contained .NET, no .NET install needed):
#   llmhub_<version>_amd64.deb         Debian, Ubuntu 24.04+ and derivatives; apt pulls in GTK 4 and libadwaita
#   LLMHub-<version>-x86_64.AppImage   any other distribution with GTK 4 and libadwaita 1.4+
#
#   desktop/tools/package/linux/package-linux.sh [--kind deb|appimage|all] [--out DIR] [--work DIR]
#
# Needs the .NET 10 SDK and dpkg-deb. The AppImage step downloads a pinned appimagetool and AppImage runtime from
# github.com/AppImage (checked against the SHA-256 below) unless APPIMAGETOOL / APPIMAGE_RUNTIME point at local copies.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
DESKTOP=$(cd "$HERE/../../.." && pwd)
KIND=all
OUT=$DESKTOP/tools/package/out
WORK=$DESKTOP/tools/package/work/linux
while [ $# -gt 0 ]; do
  case $1 in
    --kind) KIND=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

APP_ID=app.llmhub.Desktop
VERSION=$(sed -n 's/^MARKETING_VERSION *= *//p' "$DESKTOP/version.xcconfig" | tr -d '\r')
[ -n "$VERSION" ] || { echo "no MARKETING_VERSION in version.xcconfig" >&2; exit 1; }
APPIMAGETOOL_URL=https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage
APPIMAGETOOL_SHA256=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0
RUNTIME_URL=https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64
RUNTIME_SHA256=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d
mkdir -p "$OUT" "$WORK"
OUT=$(cd "$OUT" && pwd)
WORK=$(cd "$WORK" && pwd)
echo "LLM Hub $VERSION for Linux x64 ($KIND)"

# 1. Self-contained Release build.
PUBLISH=$WORK/publish
rm -rf "$PUBLISH"
dotnet publish "$DESKTOP/linux/LLMHub.Gtk/LLMHub.Gtk.csproj" -c Release -r linux-x64 --self-contained \
  -o "$PUBLISH" -nologo -v:q -clp:ErrorsOnly
rm -f "$PUBLISH"/*.pdb
chmod -R u=rwX,go=rX "$PUBLISH"
chmod 755 "$PUBLISH/llmhub" "$PUBLISH/createdump"

# The files both packages share, under a /usr-style root: the app in lib/llmhub, the launcher entry, icons, AppStream.
stage_usr() {
  local usr=$1
  mkdir -p "$usr/lib" "$usr/share/applications" "$usr/share/metainfo" \
    "$usr/share/icons/hicolor/256x256/apps" "$usr/share/icons/hicolor/512x512/apps"
  cp -a "$PUBLISH" "$usr/lib/llmhub"
  install -m 644 "$HERE/$APP_ID.desktop" "$usr/share/applications/$APP_ID.desktop"
  sed -e "s/@VERSION@/$VERSION/" -e "s/@DATE@/$(date -u +%Y-%m-%d)/" "$HERE/$APP_ID.metainfo.xml" > "$usr/share/metainfo/$APP_ID.metainfo.xml"
  chmod 644 "$usr/share/metainfo/$APP_ID.metainfo.xml"
  install -m 644 "$DESKTOP/shared/icons/llmhub.png" "$usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
  install -m 644 "$DESKTOP/shared/icons/llmhub-512.png" "$usr/share/icons/hicolor/512x512/apps/$APP_ID.png"
}

# 2. .deb: the app in /usr/lib/llmhub, `llmhub` on PATH.
if [ "$KIND" = deb ] || [ "$KIND" = all ]; then
  ROOT=$WORK/deb
  rm -rf "$ROOT"
  mkdir -p "$ROOT/DEBIAN" "$ROOT/usr/bin" "$ROOT/usr/share/doc/llmhub"
  stage_usr "$ROOT/usr"
  ln -s ../lib/llmhub/llmhub "$ROOT/usr/bin/llmhub"
  if [ -f "$DESKTOP/../LICENSE" ]; then
    install -m 644 "$DESKTOP/../LICENSE" "$ROOT/usr/share/doc/llmhub/copyright"
  fi
  SIZE=$(du -sk --exclude=DEBIAN "$ROOT" | cut -f1)
  # GTK 4, libadwaita 1.4+ (ToolbarView, NavigationSplitView); libgomp1 for llama.cpp's Linux builds; OpenSSL and ICU
  # for .NET. WebKitGTK (Design preview), audio tools (dictation, read aloud), media tools (Studio search) and git (Agent)
  # are optional: the app says what is missing.
  cat > "$ROOT/DEBIAN/control" <<EOF
Package: llmhub
Version: $VERSION
Architecture: amd64
Maintainer: LLM Hub <timmy@llm-hub.app>
Installed-Size: $SIZE
Depends: libc6 (>= 2.35), libstdc++6, libgtk-4-1, libadwaita-1-0 (>= 1.4), libgomp1, libssl3t64 | libssl3, libicu78 | libicu77 | libicu76 | libicu75 | libicu74 | libicu72 | libicu71 | libicu70, ca-certificates
Recommends: libwebkitgtk-6.0-4, libvulkan1, pipewire-bin | alsa-utils, gstreamer1.0-tools | ffmpeg, git
Section: utils
Priority: optional
Homepage: https://github.com/timmyy123/LLM-Hub
Description: Run AI models on this computer
 LLM Hub runs language models locally with llama.cpp and LiteRT-LM: chat,
 an agent, design and a studio for images, video and music, plus a local
 API in OpenAI and Anthropic format. Engines and models are downloaded on
 first use.
EOF
  DEB=$OUT/llmhub_${VERSION}_amd64.deb
  dpkg-deb --root-owner-group -Zxz --build "$ROOT" "$DEB" >/dev/null
  echo "$DEB ($(du -h "$DEB" | cut -f1))"
fi

# 3. AppImage: the same files plus AppRun, which checks for GTK 4 and libadwaita first.
if [ "$KIND" = appimage ] || [ "$KIND" = all ]; then
  fetch() { # url sha256 dest
    if [ ! -f "$3" ] || ! echo "$2  $3" | sha256sum -c --status; then
      curl -fsSL --retry 5 --retry-all-errors -o "$3.part" "$1"
      echo "$2  $3.part" | sha256sum -c --status || { echo "checksum mismatch: $1" >&2; exit 1; }
      mv "$3.part" "$3"
    fi
    chmod 755 "$3"
  }
  TOOLS=$WORK/tools
  mkdir -p "$TOOLS"
  TOOL=${APPIMAGETOOL:-$TOOLS/appimagetool-x86_64.AppImage}
  RUNTIME=${APPIMAGE_RUNTIME:-$TOOLS/runtime-x86_64}
  [ -n "${APPIMAGETOOL:-}" ] || fetch "$APPIMAGETOOL_URL" "$APPIMAGETOOL_SHA256" "$TOOL"
  [ -n "${APPIMAGE_RUNTIME:-}" ] || fetch "$RUNTIME_URL" "$RUNTIME_SHA256" "$RUNTIME"

  APPDIR=$WORK/LLMHub.AppDir
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR"
  stage_usr "$APPDIR/usr"
  # AppImage catalogues read the AppStream file under its older .appdata.xml name.
  mv "$APPDIR/usr/share/metainfo/$APP_ID.metainfo.xml" "$APPDIR/usr/share/metainfo/$APP_ID.appdata.xml"
  install -m 755 "$HERE/AppRun" "$APPDIR/AppRun"
  cp "$HERE/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
  cp "$DESKTOP/shared/icons/llmhub.png" "$APPDIR/$APP_ID.png"
  ln -s "$APP_ID.png" "$APPDIR/.DirIcon"
  # Validate offline here; appimagetool's own check also fetches every URL in the file.
  if command -v appstreamcli >/dev/null; then
    appstreamcli validate-tree --no-net "$APPDIR" >"$WORK/appstream.log" 2>&1 || { cat "$WORK/appstream.log"; exit 1; }
  fi
  IMAGE=$OUT/LLMHub-$VERSION-x86_64.AppImage
  rm -f "$IMAGE"
  # Without FUSE (containers, some CI) appimagetool still runs extracted.
  ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --no-appstream --runtime-file "$RUNTIME" "$APPDIR" "$IMAGE" \
    >"$WORK/appimagetool.log" 2>&1 || { tail -20 "$WORK/appimagetool.log"; exit 1; }
  echo "$IMAGE ($(du -h "$IMAGE" | cut -f1))"
fi
