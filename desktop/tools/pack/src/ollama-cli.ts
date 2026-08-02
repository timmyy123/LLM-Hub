import { chmod, mkdir, rm, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { execFileSync } from "node:child_process";

export const OLLAMA_BIN_ENV = "OPEN_DESIGN_OLLAMA_BIN";

type OllamaPlatform = "linux" | "mac" | "win";

/**
 * Maps platform to the GitHub release .tgz / .zip asset name that contains
 * the ollama binary. Using the versioned direct CDN URL (not /releases/latest/download/)
 * avoids the redirect-chain 404 that the un-versioned path produces.
 *
 * Mac:   ollama-darwin.tgz     → extract `ollama`
 * Win:   ollama-windows-amd64.zip → extract `ollama.exe`
 * Linux: ollama-linux-amd64.tar.zst → extract `ollama` (needs zstd or fallback)
 */
function ollamaReleaseAsset(platform: OllamaPlatform, arch: string): string {
  switch (platform) {
    case "mac":
      return "ollama-darwin.tgz";
    case "win":
      return "ollama-windows-amd64.zip";
    case "linux":
      return arch === "arm64"
        ? "ollama-linux-arm64.tar.zst"
        : "ollama-linux-amd64.tar.zst";
  }
}

function targetBinaryName(platform: OllamaPlatform): string {
  return platform === "win" ? "ollama.exe" : "ollama";
}

async function isFile(p: string): Promise<boolean> {
  try {
    return (await stat(p)).isFile();
  } catch {
    return false;
  }
}

/** Resolve the latest Ollama release tag via the GitHub API (returns e.g. "v0.32.5"). */
async function resolveLatestOllamaVersion(): Promise<string> {
  try {
    // The redirect URL from /releases/latest gives us the version tag
    const out = execFileSync(
      "curl",
      ["-fsSI", "-A", "ollama-bundler/1.0",
        "https://github.com/ollama/ollama/releases/latest"],
      { encoding: "utf8" },
    );
    const match = out.match(/location:.*\/tag\/(v[\d.]+)/i);
    if (match) return match[1];
  } catch {}

  // Fallback: GitHub API
  try {
    const out = execFileSync(
      "curl",
      ["-fsSL", "-A", "ollama-bundler/1.0",
        "https://api.github.com/repos/ollama/ollama/releases/latest"],
      { encoding: "utf8" },
    );
    const parsed = JSON.parse(out) as { tag_name?: string };
    if (parsed.tag_name) return parsed.tag_name;
  } catch {}

  // Hard-coded last-known good version as final fallback
  return "v0.32.5";
}

/** Download a URL to a local path using curl -fsSL (follows redirects). */
async function downloadWithCurl(url: string, dest: string): Promise<void> {
  await mkdir(dirname(dest), { recursive: true });
  execFileSync(
    "curl",
    ["-fsSL", "-A", "ollama-bundler/1.0", url, "-o", dest],
    { stdio: "inherit" },
  );
}

/** Extract the ollama binary from a .tgz archive. */
async function extractTgz(archivePath: string, binaryName: string, dest: string): Promise<void> {
  await mkdir(dirname(dest), { recursive: true });
  execFileSync("tar", ["-xzf", archivePath, "-C", dirname(dest), binaryName], {
    stdio: "inherit",
  });
  const extracted = join(dirname(dest), binaryName);
  if (extracted !== dest) {
    const { rename } = await import("node:fs/promises");
    await rename(extracted, dest);
  }
}

/** Extract a file from a .zip archive using unzip or PowerShell. */
async function extractZip(archivePath: string, entryName: string, dest: string): Promise<void> {
  await mkdir(dirname(dest), { recursive: true });
  try {
    execFileSync("unzip", ["-o", "-j", archivePath, entryName, "-d", dirname(dest)], {
      stdio: "inherit",
    });
    const extracted = join(dirname(dest), entryName.split("/").pop()!);
    if (extracted !== dest) {
      const { rename } = await import("node:fs/promises");
      await rename(extracted, dest);
    }
  } catch {
    execFileSync(
      "powershell",
      ["-NoProfile", "-Command",
        `Expand-Archive -Force '${archivePath}' '${dirname(dest)}'; ` +
        `Move-Item -Force '${join(dirname(dest), entryName)}' '${dest}'`],
      { stdio: "inherit" },
    );
  }
}

/**
 * Download the official Ollama binary for the target platform and place it at
 * `resourceRoot/bin/ollama[.exe]`.
 *
 * Honours the `OPEN_DESIGN_OLLAMA_BIN` env var for pre-downloaded / CI-cached
 * binaries (skip the network entirely).
 *
 * @returns `{ source, target }` on success, or `null` when skipped and
 *          `requireBundled` is false.
 */
export async function copyBundledOllamaBinary({
  env = process.env,
  platform,
  arch = process.arch,
  requireBundled = false,
  resourceRoot,
}: {
  env?: NodeJS.ProcessEnv;
  platform: OllamaPlatform;
  arch?: string;
  requireBundled?: boolean;
  resourceRoot: string;
}): Promise<{ source: string; target: string } | null> {
  // Env-var override: copy a pre-built binary straight in
  const envSource = env[OLLAMA_BIN_ENV]?.trim();
  if (envSource) {
    const target = join(resourceRoot, "bin", targetBinaryName(platform));
    await mkdir(dirname(target), { recursive: true });
    const { cp } = await import("node:fs/promises");
    await cp(envSource, target);
    if (platform !== "win") await chmod(target, 0o755);
    console.log(`[ollama-cli] Installed Ollama from ${OLLAMA_BIN_ENV}=${envSource}`);
    return { source: envSource, target };
  }

  const target = join(resourceRoot, "bin", targetBinaryName(platform));

  // Cache hit: binary already present from a prior build
  if (await isFile(target)) {
    console.log(`[ollama-cli] Bundled Ollama already present at ${target}, skipping download.`);
    return { source: target, target };
  }

  const asset = ollamaReleaseAsset(platform, arch);
  const archivePath = target + ".download" + (asset.endsWith(".zip") ? ".zip" : ".tgz");

  try {
    const version = await resolveLatestOllamaVersion();
    const url = `https://github.com/ollama/ollama/releases/download/${version}/${asset}`;
    console.log(`[ollama-cli] Downloading Ollama ${version} from ${url} …`);
    await downloadWithCurl(url, archivePath);

    if (asset.endsWith(".tgz")) {
      await extractTgz(archivePath, targetBinaryName(platform), target);
    } else {
      await extractZip(archivePath, targetBinaryName(platform), target);
    }

    await rm(archivePath, { force: true });

    if (platform !== "win") await chmod(target, 0o755);
    console.log(`[ollama-cli] Ollama bundled at ${target}`);
    return { source: target, target };
  } catch (error) {
    await rm(archivePath, { force: true }).catch(() => {});
    if (requireBundled) {
      throw new Error(
        `ollama-cli: failed to bundle Ollama for ${platform}: ${(error as Error).message}`,
        { cause: error },
      );
    }
    console.warn(
      `[ollama-cli] Warning: could not bundle Ollama — app will require user-installed Ollama. Error: ${(error as Error).message}`,
    );
    return null;
  }
}
