"""Build a deterministic Lean artifact ZIP and authenticated module index."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lean", type=pathlib.Path, required=True)
    parser.add_argument("--exporter", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--archive", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    oleans = sorted(
        path for group in ("Init", "Std", "Lean", "Lake")
        for path in [args.root / (group + ".olean"), *(args.root / group).rglob("*.olean")]
    )
    with tempfile.TemporaryDirectory(prefix="lean-module-index-") as temporary:
        file_list = pathlib.Path(temporary) / "modules.txt"
        file_list.write_text("\n".join(p.relative_to(args.root).as_posix() for p in oleans))
        result = subprocess.run(
            [str(args.lean), "--run", str(args.exporter), str(args.root), str(file_list)],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    records = [json.loads(line) for line in result.stdout.splitlines()]
    if len(records) != len(oleans):
        raise ValueError("dependency exporter did not return every module")
    modules = {}
    with zipfile.ZipFile(args.archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for olean, record in zip(oleans, records):
            files = []
            for suffix in (".olean", ".ir.sig", ".ir"):
                path = olean.with_suffix(suffix)
                data = path.read_bytes()
                relative = path.relative_to(args.root).as_posix()
                entry = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
                files.append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            if record["module"] in modules:
                raise ValueError("duplicate module: " + record["module"])
            modules[record["module"]] = {"imports": sorted(set(record["imports"])), "files": files}
    for name, module in modules.items():
        for dependency in module["imports"]:
            if dependency not in modules:
                raise ValueError(f"{name} has an unbundled dependency: {dependency}")
    with args.archive.open("rb") as archive:
        digest = hashlib.sha256()
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            digest.update(chunk)
    version = subprocess.check_output([str(args.lean), "--short-version"], text=True).strip()
    args.manifest.write_text(json.dumps({
        "schemaVersion": 1, "leanVersion": version, "archiveSHA256": digest.hexdigest(),
        "modules": modules,
    }, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Packed {len(modules)} modules: {args.archive.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
