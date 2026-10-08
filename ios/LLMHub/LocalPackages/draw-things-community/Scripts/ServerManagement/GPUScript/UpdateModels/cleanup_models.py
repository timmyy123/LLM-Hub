#!/usr/bin/env python3
"""
Script to clean up model files not present in the source of truth (NAS CSV).

This script removes files from a target server that are not listed in the
source of truth CSV file, and also cleans up corresponding entries in
the server's sha256-list.csv.

Usage:
  # Dry-run (show what would be removed)
  python3 cleanup_models.py --dry-run nas-sha256-list.csv root@server:/path/to/models

  # Actually remove files
  python3 cleanup_models.py nas-sha256-list.csv root@server:/path/to/models

  # Verbose output
  python3 cleanup_models.py --verbose nas-sha256-list.csv root@server:/path/to/models
"""

import os
import sys
import csv
import subprocess
from pathlib import Path


def parse_remote_path(path):
    """
    Parse a path that might be remote (user@host:/path) or local (/path).

    Returns:
        tuple: (is_remote, ssh_host, remote_path) or (False, None, local_path)
    """
    if ':' in path and '@' in path.split(':')[0]:
        # Format: user@host:/path
        ssh_part, remote_path = path.split(':', 1)
        return (True, ssh_part, remote_path)
    else:
        return (False, None, path)


def load_source_of_truth(csv_file):
    """
    Load filenames from source of truth CSV.

    Args:
        csv_file: Path to the source of truth CSV file

    Returns:
        set: Set of filenames that should be kept
    """
    valid_files = set()

    if not os.path.exists(csv_file):
        print(f"❌ Error: Source of truth CSV not found: {csv_file}")
        sys.exit(1)

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        # Strip whitespace from field names
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            if not row:
                continue
            filename = (row.get('filename', '') or '').strip()
            if filename:
                valid_files.add(filename)

    return valid_files


def load_csv_as_tuples(csv_file):
    """
    Load CSV file as list of (filename, sha256) tuples.

    Args:
        csv_file: Path to the CSV file

    Returns:
        list: List of (filename, sha256) tuples, sorted by filename
    """
    tuples = []

    if not os.path.exists(csv_file):
        return tuples

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            if not row:
                continue
            filename = (row.get('filename', '') or '').strip()
            sha256 = (row.get('sha256sum', '') or '').strip()
            if filename:
                tuples.append((filename, sha256))

    return sorted(tuples, key=lambda x: x[0])


def download_target_csv(ssh_host, remote_path):
    """
    Download sha256-list.csv from target server.

    Args:
        ssh_host: SSH host (user@host)
        remote_path: Remote directory path

    Returns:
        str: Local path to downloaded CSV, or None if failed
    """
    csv_path = f"{remote_path}/sha256-list.csv"
    local_temp = "/tmp/target-sha256-list-temp.csv"

    try:
        result = subprocess.run(
            ['scp', f'{ssh_host}:{csv_path}', local_temp],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return local_temp
        else:
            return None
    except Exception:
        return None


def list_remote_files(ssh_host, remote_path):
    """
    List .ckpt-tensordata and .ckpt files in a remote directory via SSH.

    Args:
        ssh_host: SSH host (user@host)
        remote_path: Path on remote system

    Returns:
        list: Sorted list of filenames
    """
    try:
        result = subprocess.run(
            ['ssh', ssh_host, f'find "{remote_path}" -maxdepth 1 -type f \\( -name "*.ckpt-tensordata" -o -name "*.ckpt" \\) -printf "%f\\n"'],
            capture_output=True,
            text=True,
            check=True
        )
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return sorted(files)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error listing remote files: {e}")
        return []


def list_local_files(directory):
    """
    List .ckpt-tensordata and .ckpt files in a local directory.

    Args:
        directory: Local directory path

    Returns:
        list: Sorted list of filenames
    """
    return sorted([
        f for f in os.listdir(directory)
        if (f.endswith('.ckpt-tensordata') or f.endswith('.ckpt'))
        and os.path.isfile(os.path.join(directory, f))
    ])


def get_file_size(file_path, ssh_host=None, remote_base_path=None):
    """
    Get file size in bytes.

    Args:
        file_path: Local file path or just filename if remote
        ssh_host: SSH host (user@host) if checking remotely
        remote_base_path: Base path on remote system

    Returns:
        int: File size in bytes, or -1 if error occurred
    """
    try:
        if ssh_host:
            full_remote_path = f"{remote_base_path}/{file_path}"
            result = subprocess.run(
                ['ssh', ssh_host, f'stat -c %s "{full_remote_path}"'],
                capture_output=True,
                text=True,
                check=True
            )
            return int(result.stdout.strip())
        else:
            return os.path.getsize(file_path)
    except (subprocess.CalledProcessError, ValueError, OSError):
        return -1


def format_size(size_bytes):
    """Format file size in human-readable format."""
    if size_bytes < 0:
        return "unknown"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def remove_files_from_server(files_to_remove, ssh_host=None, remote_path=None, local_dir=None, dry_run=False):
    """
    Remove files from server (local or remote).

    Args:
        files_to_remove: List of filenames to remove
        ssh_host: SSH host if remote
        remote_path: Remote path if remote
        local_dir: Local directory if local
        dry_run: If True, don't actually remove files

    Returns:
        int: Number of files removed
    """
    if not files_to_remove:
        return 0

    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(files_to_remove)} file(s)")
        return len(files_to_remove)

    removed_count = 0

    try:
        if ssh_host:
            # Remote files - batch remove
            files_str = ' '.join([f'"{remote_path}/{f}"' for f in files_to_remove])
            result = subprocess.run(
                ['ssh', ssh_host, f'rm -f {files_str}'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                removed_count = len(files_to_remove)
                for f in files_to_remove:
                    print(f"  ✓ Removed: {f}")
            else:
                print(f"  ✗ Error removing files: {result.stderr}")
        else:
            # Local files
            for filename in files_to_remove:
                file_path = os.path.join(local_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"  ✓ Removed: {filename}")
                    removed_count += 1
                except OSError as e:
                    print(f"  ✗ Error removing {filename}: {e}")

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")

    return removed_count


def update_sha256_csv(files_to_remove, ssh_host=None, remote_path=None, local_dir=None, dry_run=False):
    """
    Update sha256-list.csv to remove entries for deleted files.

    Args:
        files_to_remove: List of filenames that were removed
        ssh_host: SSH host if remote
        remote_path: Remote path if remote
        local_dir: Local directory if local
        dry_run: If True, don't actually modify CSV

    Returns:
        int: Number of entries removed from CSV
    """
    if not files_to_remove:
        return 0

    files_set = set(files_to_remove)

    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(files_to_remove)} entries from sha256-list.csv")
        return len(files_to_remove)

    try:
        if ssh_host:
            # Remote CSV - download, modify, upload
            csv_path = f"{remote_path}/sha256-list.csv"
            local_temp = "/tmp/sha256-list-temp.csv"

            # Download
            result = subprocess.run(
                ['scp', f'{ssh_host}:{csv_path}', local_temp],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"  ⚠️  Could not download sha256-list.csv: {result.stderr}")
                return 0

            # Read and filter
            rows_to_keep = []
            removed_count = 0
            with open(local_temp, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                for row in reader:
                    if not row:
                        continue
                    filename = (row.get('filename', '') or '').strip()
                    if filename not in files_set:
                        rows_to_keep.append(row)
                    else:
                        removed_count += 1

            # Write filtered CSV
            with open(local_temp, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'sha256sum'])
                for row in sorted(rows_to_keep, key=lambda x: x.get('filename', '')):
                    writer.writerow([row.get('filename', ''), row.get('sha256sum', '')])

            # Upload
            result = subprocess.run(
                ['scp', local_temp, f'{ssh_host}:{csv_path}'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"  ✓ Updated sha256-list.csv (removed {removed_count} entries)")
            else:
                print(f"  ✗ Error uploading sha256-list.csv: {result.stderr}")

            # Cleanup temp file
            os.remove(local_temp)
            return removed_count

        else:
            # Local CSV
            csv_path = os.path.join(local_dir, 'sha256-list.csv')
            if not os.path.exists(csv_path):
                print(f"  ⚠️  sha256-list.csv not found")
                return 0

            # Read and filter
            rows_to_keep = []
            removed_count = 0
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                for row in reader:
                    if not row:
                        continue
                    filename = (row.get('filename', '') or '').strip()
                    if filename not in files_set:
                        rows_to_keep.append(row)
                    else:
                        removed_count += 1

            # Write filtered CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'sha256sum'])
                for row in sorted(rows_to_keep, key=lambda x: x.get('filename', '')):
                    writer.writerow([row.get('filename', ''), row.get('sha256sum', '')])

            print(f"  ✓ Updated sha256-list.csv (removed {removed_count} entries)")
            return removed_count

    except Exception as e:
        print(f"  ✗ Error updating sha256-list.csv: {e}")
        return 0


def main():
    # Parse arguments
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    args = [arg for arg in sys.argv[1:] if arg not in ['--dry-run', '--verbose', '-v']]

    if len(args) != 2:
        print("Usage: python3 cleanup_models.py [--dry-run] [--verbose] <source_of_truth.csv> <target_server>")
        print()
        print("Options:")
        print("  --dry-run    Show what would be removed without actually removing")
        print("  --verbose    Show detailed information including file sizes")
        print()
        print("Arguments:")
        print("  source_of_truth.csv  CSV file containing the list of valid files (e.g., nas-sha256-list.csv)")
        print("  target_server        Target server to clean up")
        print("                       Local: /path/to/models")
        print("                       Remote: root@hostname:/path/to/models")
        print()
        print("Examples:")
        print("  # Dry-run to see what would be removed")
        print("  python3 cleanup_models.py --dry-run nas-sha256-list.csv root@dfw-026-001:/mnt/models/official-models")
        print()
        print("  # Actually remove files")
        print("  python3 cleanup_models.py nas-sha256-list.csv root@dfw-026-001:/mnt/models/official-models")
        sys.exit(1)

    source_csv = args[0]
    target = args[1]

    # Parse target path
    is_remote, ssh_host, target_path = parse_remote_path(target)

    # Header
    print("=" * 70)
    print("Model Cleanup Script")
    if dry_run:
        print("[DRY RUN MODE - No files will be removed]")
    print("=" * 70)
    print(f"Source of truth: {source_csv}")
    print(f"Target: {target}")
    print()

    # Step 1: Load source of truth
    print("📋 Loading source of truth CSV...")
    valid_files = load_source_of_truth(source_csv)
    print(f"   ✅ Found {len(valid_files)} valid files in source of truth")

    # Step 2: List files on target
    print(f"\n📂 Listing files on target...")
    if is_remote:
        target_files = list_remote_files(ssh_host, target_path)
    else:
        if not os.path.isdir(target_path):
            print(f"❌ Error: {target_path} is not a valid directory")
            sys.exit(1)
        target_files = list_local_files(target_path)
    print(f"   ✅ Found {len(target_files)} files on target")

    # Step 3: Load target CSV to find orphan entries
    print(f"\n📂 Loading target sha256-list.csv...")
    if is_remote:
        target_csv_local = download_target_csv(ssh_host, target_path)
    else:
        target_csv_local = os.path.join(target_path, 'sha256-list.csv')

    target_csv_entries = set()
    if target_csv_local and os.path.exists(target_csv_local):
        target_tuples = load_csv_as_tuples(target_csv_local)
        target_csv_entries = {t[0] for t in target_tuples}
        print(f"   ✅ Found {len(target_csv_entries)} entries in target CSV")
    else:
        print(f"   ⚠️  Could not load target sha256-list.csv")

    # Step 4: Find files to remove (on disk but not in source of truth)
    target_files_set = set(target_files)
    files_to_remove = [f for f in target_files if f not in valid_files]

    # Find orphan CSV entries (in CSV but not on disk)
    orphan_csv_entries = [f for f in target_csv_entries if f not in target_files_set]

    # Total entries to remove from CSV = files_to_remove + orphan_csv_entries
    csv_entries_to_remove = set(files_to_remove) | set(orphan_csv_entries)

    files_remaining = len(target_files) - len(files_to_remove)

    print(f"\n📊 Analysis:")
    print(f"   Files in source of truth:  {len(valid_files)}")
    print(f"   Files on target (before):  {len(target_files)}")
    print(f"   Files to remove:           {len(files_to_remove)}")
    print(f"   Orphan CSV entries:        {len(orphan_csv_entries)}")
    print(f"   Files remaining (after):   {files_remaining}")

    if not files_to_remove:
        print("\n✅ No files to remove - target is clean!")
        sys.exit(0)

    # Step 4: Show files to be removed
    print(f"\n{'=' * 70}")
    print(f"Files to be removed ({len(files_to_remove)}):")
    print("=" * 70)

    total_size = 0
    for i, filename in enumerate(files_to_remove, 1):
        if verbose:
            if is_remote:
                size = get_file_size(filename, ssh_host, target_path)
            else:
                size = get_file_size(os.path.join(target_path, filename))
            total_size += max(0, size)
            print(f"  {i:4}. {filename} ({format_size(size)})")
        else:
            print(f"  {i:4}. {filename}")

    if verbose and total_size > 0:
        print(f"\n   Total size to be freed: {format_size(total_size)}")

    # Step 5: Confirm and remove
    if dry_run:
        # Download and verify sha256-list.csv from target
        print(f"\n{'=' * 70}")
        print("[DRY RUN] Verifying cleanup result...")
        print("=" * 70)

        # Load source of truth as tuples
        source_tuples = load_csv_as_tuples(source_csv)
        source_dict = {t[0]: t[1] for t in source_tuples}
        print(f"   Source of truth entries: {len(source_tuples)}")

        # Download target CSV
        if is_remote:
            target_csv_local = download_target_csv(ssh_host, target_path)
        else:
            target_csv_local = os.path.join(target_path, 'sha256-list.csv')

        if target_csv_local and os.path.exists(target_csv_local):
            target_tuples = load_csv_as_tuples(target_csv_local)
            print(f"   Target CSV entries (before): {len(target_tuples)}")

            # Simulate removal - filter out files_to_remove AND orphan entries
            simulated_remaining = [(f, sha) for f, sha in target_tuples if f not in csv_entries_to_remove]
            print(f"   Target CSV entries (after):  {len(simulated_remaining)}")

            # Compare with source of truth
            print(f"\n   Verification:")

            # Check 1: All remaining files should exist in source of truth
            missing_in_source = []
            checksum_mismatch = []
            matching = []

            for filename, target_sha in simulated_remaining:
                if filename not in source_dict:
                    missing_in_source.append(filename)
                elif source_dict[filename] and target_sha and source_dict[filename] != target_sha:
                    checksum_mismatch.append((filename, target_sha, source_dict[filename]))
                else:
                    matching.append(filename)

            print(f"   - Files matching source:     {len(matching)}")

            if missing_in_source:
                print(f"   - ⚠️  Files not in source:   {len(missing_in_source)}")
                if verbose:
                    for f in missing_in_source[:5]:
                        print(f"        {f}")
                    if len(missing_in_source) > 5:
                        print(f"        ... and {len(missing_in_source) - 5} more")

            if checksum_mismatch:
                print(f"   - ⚠️  Checksum mismatches:   {len(checksum_mismatch)}")
                if verbose:
                    for f, t_sha, s_sha in checksum_mismatch[:3]:
                        print(f"        {f}")
                        print(f"          target: {t_sha[:16]}...")
                        print(f"          source: {s_sha[:16]}...")
                    if len(checksum_mismatch) > 3:
                        print(f"        ... and {len(checksum_mismatch) - 3} more")

            # Final verification result
            if not missing_in_source and not checksum_mismatch:
                print(f"\n   ✅ Verification PASSED: After cleanup, target will match source of truth")
            else:
                print(f"\n   ⚠️  Verification WARNING: Some discrepancies detected")
                print(f"      (This may be expected if source has files not yet synced to target)")

            # Cleanup temp file if remote
            if is_remote and target_csv_local and os.path.exists(target_csv_local):
                os.remove(target_csv_local)
        else:
            print(f"   ⚠️  Could not download target sha256-list.csv for verification")

        print(f"\n{'=' * 70}")
        print("[DRY RUN] Summary")
        print("=" * 70)
        print(f"   Files to remove:           {len(files_to_remove)}")
        print(f"   Orphan CSV entries:        {len(orphan_csv_entries)}")
        print(f"   Total CSV entries to remove: {len(csv_entries_to_remove)}")
        print(f"   Files remaining (after):   {files_remaining}")
        print("=" * 70)
        print("\nTo actually remove these files, run without --dry-run flag")
        sys.exit(0)

    # Actually remove files
    print(f"\n{'=' * 70}")
    print("Removing files...")
    print("=" * 70)

    removed_count = remove_files_from_server(
        files_to_remove,
        ssh_host=ssh_host,
        remote_path=target_path,
        local_dir=target_path if not is_remote else None,
        dry_run=False
    )

    # Update sha256-list.csv (remove both deleted files and orphan entries)
    print(f"\n{'=' * 70}")
    print("Updating sha256-list.csv...")
    print("=" * 70)

    if orphan_csv_entries:
        print(f"   Also removing {len(orphan_csv_entries)} orphan CSV entries (in CSV but not on disk)")

    csv_removed = update_sha256_csv(
        list(csv_entries_to_remove),  # includes files_to_remove + orphan_csv_entries
        ssh_host=ssh_host,
        remote_path=target_path,
        local_dir=target_path if not is_remote else None,
        dry_run=False
    )

    # Summary
    actual_remaining = len(target_files) - removed_count

    print(f"\n{'=' * 70}")
    print("Summary")
    print("=" * 70)
    print(f"   Files removed:             {removed_count}/{len(files_to_remove)}")
    print(f"   CSV entries removed:       {csv_removed}")
    print(f"   Files remaining on target: {actual_remaining}")
    print("=" * 70)

    if removed_count == len(files_to_remove):
        print("\n✅ Cleanup completed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some files could not be removed")
        sys.exit(1)


if __name__ == '__main__':
    main()
