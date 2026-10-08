"""Retain Lean's name-only interpreter ABI without an app-wide export policy."""

import argparse
import fnmatch
from pathlib import Path
import subprocess


def public_symbols(nm_output, patterns):
    symbols = set()
    for line in nm_output.splitlines():
        if " external " not in line or " private external " in line:
            continue
        if "(undefined)" in line:
            continue
        symbol = line.split()[-1]
        if any(fnmatch.fnmatchcase(symbol, pattern) for pattern in patterns):
            symbols.add(symbol)
    # These are renamed process-terminating implementations, not the public
    # embedding-safe replacements in LeanPlatformStubs.cpp.
    symbols.difference_update(
        {"_lean_native_process_exit", "_lean_native_process_force_exit"}
    )
    return symbols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("archives", nargs="+")
    args = parser.parse_args()
    patterns = [
        line.strip()
        for line in args.allowlist.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    symbols = set()
    for archive in args.archives:
        output = subprocess.check_output(["xcrun", "nm", "-m", archive], text=True)
        symbols.update(public_symbols(output, patterns))
    required = {
        "_lean_io_exit",
        "_lean_io_force_exit",
        "_lean_openssl_version",
        "_lean_run_mod_init_core",
        "_runtime_initialize_Init",
        "_meta_initialize_Lean",
        "_runtime_initialize_Std",
    }
    if missing := required - symbols:
        raise RuntimeError(f"missing Lean interpreter ABI: {sorted(missing)}")
    ordered = sorted(symbols)
    args.symbols.write_text("".join(f"{symbol}\n" for symbol in ordered))
    # -u roots survive archive extraction, LTO, and dead stripping. A linker
    # response file avoids ARG_MAX. Unlike -exported_symbols_list, this does not
    # hide other runtimes' exports; unlike -export_dynamic, it retains only the
    # selected Lean ABI. Default-visible roots remain available to dlsym after
    # stripping local/debug symbols.
    args.response.write_text("".join(f"-u\n{symbol}\n" for symbol in ordered))


if __name__ == "__main__":
    main()
