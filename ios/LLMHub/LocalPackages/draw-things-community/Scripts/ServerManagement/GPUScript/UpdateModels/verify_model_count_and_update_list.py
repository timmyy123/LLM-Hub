#!/usr/bin/env python3
"""
Verify that all GPU servers have the same number of models (.ckpt and .ckpt-tensordata).
Reads server list from gpu_servers.csv, compares model counts, filters the verified
.ckpt names through model_blacklist.txt, writes model-list, and optionally triggers
a control panel update.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# Paths derived from script location
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parents[3]  # Scripts/ServerManagement/GPUScript/UpdateModels -> repo root
SERVERS_FILE = SCRIPT_DIR / "gpu_servers_logic.csv"
MODEL_BLACKLIST_FILE = SCRIPT_DIR / "model_blacklist.txt"
MODEL_LIST_FILE = REPO_ROOT / "model-list"
SSH_TIMEOUT = 30

# Control panel settings
CONTROL_PANEL_HOST = "100.71.37.12"
CONTROL_PANEL_PORT = "50002"


def parse_servers(file_path: Path) -> list[tuple[str, str]]:
    """Parse server list from CSV file, returning (user@hostname, models_path) tuples."""
    servers = []
    with open(file_path, "r", newline="") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Parse CSV row: remote_host, models_path [, nas_url]
            reader = csv.reader([line])
            row = next(reader)
            row = [col.strip() for col in row]
            if len(row) >= 2:
                remote_host = row[0]
                models_path = row[1]
                servers.append((remote_host, models_path))
    return servers


def get_file_count(server: str, models_path: str, pattern: str) -> tuple[int | None, str]:
    """
    SSH into server and count files matching pattern.
    Returns (count, error_message). count is None if error occurred.
    """
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        server,
        f"ls -lrta {models_path} 2>/dev/null | awk '/{pattern}$/ {{print $NF}}' | sort | wc -l"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or "SSH connection failed"

        count = int(result.stdout.strip())
        return count, ""
    except subprocess.TimeoutExpired:
        return None, "Connection timed out"
    except ValueError as e:
        return None, f"Failed to parse count: {e}"
    except Exception as e:
        return None, str(e)


def get_file_list(server: str, models_path: str, pattern: str) -> tuple[list[str] | None, str]:
    """
    SSH into server and get a sorted list of names matching pattern.
    Returns (file_list, error_message). file_list is None if error occurred.
    """
    cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        server,
        f"ls -lrta {models_path} 2>/dev/null | awk '/{pattern}$/ {{print $NF}}' | sort"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT
        )
        if result.returncode != 0:
            return None, result.stderr.strip() or "SSH connection failed"

        models = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return models, ""
    except subprocess.TimeoutExpired:
        return None, "Connection timed out"
    except Exception as e:
        return None, str(e)


def update_control_panel(model_list_path: Path) -> tuple[bool, str]:
    """
    Run bazel command to update control panel with model list.
    Returns (success, output_or_error).
    """
    cmd = [
        "bazel", "run", "Apps:ProxyServerControlPanelCLI", "--",
        "-h", CONTROL_PANEL_HOST,
        "-p", CONTROL_PANEL_PORT,
        "update-model-list", str(model_list_path)
    ]

    print(f"  Command: {' '.join(cmd)}")
    print("  (This may take a few minutes to build...)")

    try:
        # Stream output in real-time instead of capturing
        result = subprocess.run(
            cmd,
            text=True,
            cwd=REPO_ROOT,
            timeout=300  # 5 minutes for bazel build + run
        )
        if result.returncode != 0:
            return False, f"Command failed with exit code {result.returncode}"
        return True, "Success"
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 5 minutes"
    except Exception as e:
        return False, str(e)


def verify_counts(servers: list[tuple[str, str]]) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Verify .ckpt and .ckpt-tensordata counts across all servers.
    Returns (results, errors) where results maps server -> {ckpt: count, tensordata: count, models_path: path}
    """
    results = {}
    errors = {}

    for server, models_path in servers:
        print(f"Checking {server} ({models_path})...", end=" ", flush=True)

        ckpt_count, ckpt_error = get_file_count(server, models_path, "\\.ckpt")
        if ckpt_error:
            errors[server] = ckpt_error
            print(f"ERROR: {ckpt_error}")
            continue

        tensordata_count, tensordata_error = get_file_count(server, models_path, "\\.ckpt-tensordata")
        if tensordata_error:
            errors[server] = tensordata_error
            print(f"ERROR: {tensordata_error}")
            continue

        results[server] = {
            "ckpt": ckpt_count,
            "tensordata": tensordata_count,
            "models_path": models_path
        }
        print(f".ckpt: {ckpt_count}, .ckpt-tensordata: {tensordata_count}")

    return results, errors


def check_consistency(results: dict[str, dict]) -> tuple[bool, str]:
    """
    Check if all servers have consistent counts.
    Returns (is_consistent, summary_message).
    """
    if not results:
        return False, "No results to check"

    ckpt_counts = {s: r["ckpt"] for s, r in results.items()}
    tensordata_counts = {s: r["tensordata"] for s, r in results.items()}

    unique_ckpt = set(ckpt_counts.values())
    unique_tensordata = set(tensordata_counts.values())

    messages = []
    consistent = True

    if len(unique_ckpt) == 1:
        messages.append(f"All servers have {list(unique_ckpt)[0]} .ckpt files")
    else:
        consistent = False
        messages.append(".ckpt count mismatch:")
        count_to_servers = {}
        for server, count in ckpt_counts.items():
            count_to_servers.setdefault(count, []).append(server)
        for count, server_list in sorted(count_to_servers.items(), reverse=True):
            messages.append(f"  {count}: {', '.join(server_list)}")

    if len(unique_tensordata) == 1:
        messages.append(f"All servers have {list(unique_tensordata)[0]} .ckpt-tensordata files")
    else:
        consistent = False
        messages.append(".ckpt-tensordata count mismatch:")
        count_to_servers = {}
        for server, count in tensordata_counts.items():
            count_to_servers.setdefault(count, []).append(server)
        for count, server_list in sorted(count_to_servers.items(), reverse=True):
            messages.append(f"  {count}: {', '.join(server_list)}")

    return consistent, "\n".join(messages)


def print_file_differences(results: dict[str, dict]) -> None:
    """Print filename differences for each count mismatch."""
    if not results:
        return

    for count_key, pattern, label in (
        ("ckpt", "\\.ckpt", ".ckpt"),
        ("tensordata", "\\.ckpt-tensordata", ".ckpt-tensordata"),
    ):
        count_to_servers = {}
        for server, info in results.items():
            count_to_servers.setdefault(info[count_key], []).append(server)
        if len(count_to_servers) == 1:
            continue

        # Use the most common count as the reference so an outlier with one
        # extra file is reported as extra instead of every other server missing it.
        reference_count = max(count_to_servers, key=lambda count: len(count_to_servers[count]))
        reference_server = count_to_servers[reference_count][0]
        reference_info = results[reference_server]
        reference_path = reference_info["models_path"]

        print(
            f"\nFetching {label} list from reference: "
            f"{reference_server} ({reference_path})..."
        )
        reference_files, reference_error = get_file_list(
            reference_server, reference_path, pattern
        )
        if reference_error:
            print(f"  Error: {reference_error}")
            continue
        reference_set = set(reference_files)

        for server, info in results.items():
            if info[count_key] == reference_count:
                continue

            server_path = info["models_path"]
            print(f"\nFetching {label} list from {server} ({server_path})...")
            server_files, server_error = get_file_list(server, server_path, pattern)
            if server_error:
                print(f"  Error: {server_error}")
                continue
            server_set = set(server_files)

            missing = sorted(reference_set - server_set)
            extra = sorted(server_set - reference_set)

            if missing:
                print(f"  Missing {label} on {server} ({len(missing)}):")
                for filename in missing:
                    print(f"    - {filename}")
            if extra:
                print(f"  Extra {label} on {server} ({len(extra)}):")
                for filename in extra:
                    print(f"    + {filename}")
            if not missing and not extra:
                print(f"  No {label} name differences found")


def main():
    parser = argparse.ArgumentParser(
        description="Verify model counts across GPU servers and optionally update control panel"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update control panel after successful verification"
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Update control panel even if verification fails"
    )
    args = parser.parse_args()

    if not SERVERS_FILE.exists():
        print(f"Error: Server file not found: {SERVERS_FILE}")
        sys.exit(1)

    servers = parse_servers(SERVERS_FILE)
    if not servers:
        print("Error: No servers found in config file")
        sys.exit(1)

    if not MODEL_BLACKLIST_FILE.exists():
        print(f"Error: Model blacklist file not found: {MODEL_BLACKLIST_FILE}")
        sys.exit(1)

    try:
        with open(MODEL_BLACKLIST_FILE, "r", encoding="utf-8") as f:
            model_blacklist = {
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
    except OSError as e:
        print(f"Error reading model blacklist: {e}")
        sys.exit(1)

    invalid_blacklist_entries = sorted(
        entry
        for entry in model_blacklist
        if not entry.endswith(".ckpt") or Path(entry).name != entry
    )
    if invalid_blacklist_entries:
        print(
            "Error: Model blacklist entries must be exact .ckpt filenames "
            "without directory paths:"
        )
        for entry in invalid_blacklist_entries:
            print(f"  - {entry}")
        sys.exit(1)

    print(f"Repository root: {REPO_ROOT}")
    print(
        f"Loaded {len(model_blacklist)} model blacklist entries from "
        f"{MODEL_BLACKLIST_FILE}"
    )
    print(f"Checking model counts on {len(servers)} servers...")
    print("-" * 70)

    results, errors = verify_counts(servers)

    print("-" * 70)

    # Report errors
    if errors:
        print(f"\n⚠ {len(errors)} server(s) failed to connect:")
        for server, error in errors.items():
            print(f"  - {server}: {error}")

    if not results:
        print("\nNo successful connections. Cannot verify model counts.")
        sys.exit(1)

    # Check consistency
    consistent, summary = check_consistency(results)
    print(f"\n{'✓' if consistent else '✗'} Verification {'passed' if consistent else 'failed'}:")
    print(summary)

    if not consistent:
        print_file_differences(results)

    # Determine if we should update
    should_update = args.force_update or (args.update and consistent and not errors)

    if args.update and not consistent:
        print("\n⚠ Skipping control panel update due to verification failure")
        print("  Use --force-update to update anyway")

    if args.update and errors and consistent:
        print("\n⚠ Skipping control panel update due to connection errors")
        print("  Use --force-update to update anyway")

    # A normal successful verification should still produce the model list.
    # --force-update preserves the existing behavior of using the first
    # successfully queried server even when verification is incomplete.
    should_save_model_list = (consistent and not errors) or should_update
    if should_save_model_list:
        first_server = next(iter(results))
        first_server_path = results[first_server]["models_path"]
        print(f"\nFetching .ckpt list from {first_server} ({first_server_path})...")

        model_list, error = get_file_list(first_server, first_server_path, "\\.ckpt")
        if error or model_list is None:
            print(f"Error fetching .ckpt list: {error or 'No file list returned'}")
            sys.exit(1)

        expected_count = results[first_server]["ckpt"]
        if len(model_list) != expected_count:
            print(
                "Error: .ckpt count changed while fetching the model list "
                f"(expected {expected_count}, got {len(model_list)})"
            )
            sys.exit(1)

        filtered_model_list = [
            model for model in model_list if model not in model_blacklist
        ]
        excluded_models = sorted(set(model_list) & model_blacklist)
        unmatched_blacklist_entries = sorted(model_blacklist - set(model_list))

        print(
            f"Excluded {len(excluded_models)} blacklisted .ckpt files; "
            f"{len(filtered_model_list)} remain"
        )
        if unmatched_blacklist_entries:
            print(
                f"⚠ {len(unmatched_blacklist_entries)} blacklist entries were not "
                "present on the source server"
            )

        print(
            f"Saving {len(filtered_model_list)} filtered .ckpt files to "
            f"{MODEL_LIST_FILE}..."
        )
        with open(MODEL_LIST_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_model_list))
            if filtered_model_list:
                f.write("\n")
        print(f"✓ Model list saved to {MODEL_LIST_FILE}")

    if should_update:
        print("\n" + "=" * 70)
        print("Updating control panel...")

        # Run bazel command
        print(f"Running control panel update...")
        print(f"  Host: {CONTROL_PANEL_HOST}:{CONTROL_PANEL_PORT}")
        success, output = update_control_panel(MODEL_LIST_FILE)

        if success:
            print("✓ Control panel updated successfully")
        else:
            print(f"✗ Control panel update failed:\n{output}")
            sys.exit(1)

    if not consistent or errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
