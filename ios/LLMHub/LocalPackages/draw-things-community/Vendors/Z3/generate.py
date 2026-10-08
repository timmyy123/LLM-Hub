#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


_EXCLUDED_SCAN_DIRECTORIES = (
    "src/api/dotnet/",
    "src/api/java/",
    "src/api/ml/",
    "src/api/python/",
    "src/shell/",
    "src/test/",
)

_API_HEADERS = (
    "z3_api.h",
    "z3_ast_containers.h",
    "z3_algebraic.h",
    "z3_polynomial.h",
    "z3_rcf.h",
    "z3_fixedpoint.h",
    "z3_optimization.h",
    "z3_fpa.h",
    "z3_spacer.h",
)


def _is_scanned_header(source_root: Path, path: Path) -> bool:
    relative_path = path.relative_to(source_root).as_posix()
    return not relative_path.startswith(_EXCLUDED_SCAN_DIRECTORIES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-file", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    version_file = Path(args.version_file).resolve()
    source_root = version_file.parent.parent
    output_root = Path(args.output_root).resolve()
    generated_source_root = output_root / "src"

    sys.path.insert(0, str(source_root / "scripts"))
    import mk_genfile_common
    import update_api

    generated_headers = []
    for pyg_path in sorted((source_root / "src").rglob("*.pyg")):
        relative_directory = pyg_path.parent.relative_to(source_root)
        output_directory = output_root / relative_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        generated_headers.append(
            Path(mk_genfile_common.mk_hpp_from_pyg(str(pyg_path), str(output_directory)))
        )

    pattern_output = generated_source_root / "ast/pattern/database.h"
    pattern_output.parent.mkdir(parents=True, exist_ok=True)
    mk_genfile_common.mk_pat_db_internal(
        str(source_root / "src/ast/pattern/database.smt2"),
        str(pattern_output),
    )
    generated_headers.append(pattern_output)

    api_output_directory = generated_source_root / "api"
    api_output_directory.mkdir(parents=True, exist_ok=True)
    update_api.VERBOSE = False
    update_api.generate_files(
        api_files=[str(source_root / "src/api" / name) for name in _API_HEADERS],
        api_output_dir=str(api_output_directory),
    )
    generated_headers.append(api_output_directory / "api_log_macros.h")

    version = version_file.read_text(encoding="utf-8").strip().split(".")
    if len(version) not in (3, 4):
        raise ValueError(f"unexpected Z3 version: {'.'.join(version)}")
    revision = version[3] if len(version) == 4 else "0"
    version_header = generated_source_root / "util/z3_version.h"
    version_header.parent.mkdir(parents=True, exist_ok=True)
    version_header.write_text(
        "// automatically generated file.\n"
        f"#define Z3_MAJOR_VERSION   {version[0]}\n"
        f"#define Z3_MINOR_VERSION   {version[1]}\n"
        f"#define Z3_BUILD_NUMBER    {version[2]}\n"
        f"#define Z3_REVISION_NUMBER {revision}\n\n"
        f'#define Z3_FULL_VERSION    "Z3 {version[0]}.{version[1]}.{version[2]}.{revision}"\n',
        encoding="utf-8",
    )
    generated_headers.append(version_header)

    source_headers = sorted(
        path
        for pattern in ("*.h", "*.hpp")
        for path in (source_root / "src").rglob(pattern)
        if _is_scanned_header(source_root, path)
    )
    scanned_headers = [str(path) for path in source_headers + generated_headers]
    dll_output_directory = generated_source_root / "api/dll"
    dll_output_directory.mkdir(parents=True, exist_ok=True)
    mk_genfile_common.mk_install_tactic_cpp_internal(
        scanned_headers, str(dll_output_directory)
    )
    mk_genfile_common.mk_mem_initializer_cpp_internal(
        scanned_headers, str(dll_output_directory)
    )
    mk_genfile_common.mk_gparams_register_modules_internal(
        scanned_headers, str(dll_output_directory)
    )


if __name__ == "__main__":
    main()
