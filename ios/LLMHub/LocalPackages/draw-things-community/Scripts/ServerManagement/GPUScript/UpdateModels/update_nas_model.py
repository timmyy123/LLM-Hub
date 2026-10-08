#!/usr/bin/env python3
"""
NAS Model Update Script

Usage:
  Local:  python update_nas_model.py -p /path/to/models -u /path/to/utils/UpdateModels -m MODE --account-id ID --access-key KEY --secret-key SECRET
  Remote: python update_nas_model.py root@hostname -p /path/to/models -u /path/to/utils/ -r /path/to/draw-things UpdateModels -m MODE --account-id ID --access-key KEY --secret-key SECRET

Required parameters:
  -p, --path          Path to models directory (e.g., /zfs/data/official-models-ckpt-tensordata/)
  -u, --utils         Path to utils directory (e.g., /root/utils/UpdateModels)
  -r, --repo-path     Path to draw-things repository (default: {utils}/draw-things)
  -m, --mode          Operation mode: sync, format, checksum, or all (default: all)
  --account-id        Cloudflare R2 account ID (required for sync mode)
  --access-key        R2 access key (required for sync mode)
  --secret-key        R2 secret key (required for sync mode)
  --dry-run           Show what would be done without making any changes
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)


def print_separator() -> None:
    """Print a separator line."""
    print("=" * 50)


def run_command(cmd: list[str], check: bool = True, dry_run: bool = False) -> subprocess.CompletedProcess | None:
    """Run a command and return the result."""
    if dry_run:
        print(f"  [dry-run] Would execute: {' '.join(cmd)}")
        return None
    return subprocess.run(cmd, check=check)


def validate_path_exists(path: str, description: str) -> None:
    """Validate that a path exists."""
    if not os.path.isdir(path):
        print(f"Error: {description} does not exist: {path}")
        sys.exit(1)


def validate_file_exists(path: str, description: str) -> None:
    """Validate that a file exists."""
    if not os.path.isfile(path):
        print(f"Error: {description} not found: {path}")
        sys.exit(1)


def run_local(
    models_path: str,
    utils_path: str,
    repo_path: str,
    mode: str,
    account_id: str | None,
    access_key: str | None,
    secret_key: str | None,
    dry_run: bool = False,
    exclude: str | None = None,
) -> None:
    """Run operations locally."""
    print_header("Local NAS Model Update" + (" (DRY RUN)" if dry_run else ""))
    print(f"Models path: {models_path}")
    print(f"Utils path:  {utils_path}")
    print(f"Repo path:   {repo_path}")
    print(f"Mode:        {mode}")
    if dry_run:
        print("Dry run:     Yes (no changes will be made)")
    print()

    # Validate paths exist
    validate_path_exists(models_path, "Models path")
    validate_path_exists(utils_path, "Utils path")
    validate_path_exists(repo_path, "Repository path")

    # Step 1: R2 Sync Verification
    if mode in ("all", "sync"):
        print("➡️  Step 1: Running R2 sync verification...")
        print_separator()

        r2_script = os.path.join(utils_path, "r2_sync_verification.py")
        validate_file_exists(r2_script, "R2 sync script")

        cmd = [
            "python3", r2_script,
            "-p", models_path,
            "--account-id", account_id,
            "--access-key", access_key,
            "--secret-key", secret_key,
        ]
        if exclude:
            cmd += ["--exclude", exclude]
        run_command(cmd, dry_run=dry_run)

        print("✅ R2 sync verification completed")
        print()

    # Step 2: Update repository and run TensorDataFormatter
    if mode in ("all", "format"):
        print("➡️  Step 2: Updating draw-things repository...")
        print_separator()

        if dry_run:
            print(f"  [dry-run] Would run in {repo_path}:")
            run_command(["git", "pull", "origin", "main"], dry_run=dry_run)
        else:
            subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=repo_path,
                check=True,
            )

        print("✅ Repository updated")
        print()

        print("➡️  Step 3: Running TensorDataFormatter...")
        print_separator()

        if dry_run:
            print(f"  [dry-run] Would run in {repo_path}:")
            run_command(["bazel", "run", "Apps:TensorDataFormatter", "-c", "opt", "--", "--path", models_path], dry_run=dry_run)
        else:
            subprocess.run(
                ["bazel", "run", "Apps:TensorDataFormatter", "-c", "opt", "--", "--path", models_path],
                cwd=repo_path,
                check=True,
            )

        print("✅ TensorData formatting completed")
        print()

    # Step 4: Checksum Comparison
    if mode in ("all", "checksum"):
        print("➡️  Step 4: Comparing checksums...")
        print_separator()

        checksum_script = os.path.join(utils_path, "compare_checksums.py")
        validate_file_exists(checksum_script, "Checksum comparison script")

        run_command(["python3", checksum_script, models_path], dry_run=dry_run)

        print("✅ Checksum comparison completed")
        print()

    print_separator()
    print("✅ NAS Model Update Complete!" + (" (DRY RUN)" if dry_run else ""))
    print_separator()


def run_ssh_command(remote_host: str, cmd: str, dry_run: bool = False) -> None:
    """Run a command on remote server via SSH."""
    if dry_run:
        print(f"  [dry-run] ssh {remote_host} '{cmd}'")
    else:
        subprocess.run(["ssh", remote_host, cmd], check=True)


def run_remote(
    remote_host: str,
    models_path: str,
    utils_path: str,
    repo_path: str,
    mode: str,
    account_id: str | None,
    access_key: str | None,
    secret_key: str | None,
    dry_run: bool = False,
    exclude: str | None = None,
) -> None:
    """Run operations on a remote server via SSH."""
    print_header("Remote NAS Model Update" + (" (DRY RUN)" if dry_run else ""))
    print(f"Target:      {remote_host}")
    print(f"Models path: {models_path}")
    print(f"Utils path:  {utils_path}")
    print(f"Repo path:   {repo_path}")
    print(f"Mode:        {mode}")
    if dry_run:
        print("Dry run:     Yes (no changes will be made)")
    print()

    # Check if ssh is available
    if not shutil.which("ssh"):
        print("Error: ssh command not found. Please install openssh-client.")
        sys.exit(1)

    # Test SSH connection (skip in dry-run mode)
    if dry_run:
        print(f"[dry-run] Would test SSH connection to {remote_host}")
    else:
        print(f"Testing SSH connection to {remote_host}...")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", remote_host, "echo 'Connection successful'"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error: Cannot connect to {remote_host}")
            print("Please ensure:")
            print("  1. The host is reachable")
            print("  2. SSH keys are set up for passwordless authentication")
            print("  3. You have appropriate access")
            sys.exit(1)
        print("✅ SSH connection successful")
    print()

    # Step 1: R2 Sync Verification
    if mode in ("all", "sync"):
        print("➡️  Step 1: Running R2 sync verification...")
        print_separator()

        r2_script = f"{utils_path}/r2_sync_verification.py"
        r2_cmd = f"python3 {r2_script} -p {models_path} --account-id {account_id} --access-key {access_key} --secret-key {secret_key}"
        if exclude:
            r2_cmd += f" --exclude {exclude}"
        run_ssh_command(remote_host, r2_cmd, dry_run=dry_run)

        print("✅ R2 sync verification completed")
        print()

    # Step 2: Update repository
    if mode in ("all", "format"):
        print("➡️  Step 2: Updating draw-things repository...")
        print_separator()

        git_cmd = f"cd {repo_path} && git pull origin main"
        run_ssh_command(remote_host, git_cmd, dry_run=dry_run)

        print("✅ Repository updated")
        print()

        # Step 3: Run TensorDataFormatter
        print("➡️  Step 3: Running TensorDataFormatter...")
        print_separator()

        bazel_cmd = f"cd {repo_path} && bazel run Apps:TensorDataFormatter -c opt -- --path {models_path}"
        run_ssh_command(remote_host, bazel_cmd, dry_run=dry_run)

        print("✅ TensorData formatting completed")
        print()

    # Step 4: Checksum Comparison
    if mode in ("all", "checksum"):
        print("➡️  Step 4: Comparing checksums...")
        print_separator()

        checksum_script = f"{utils_path}/compare_checksums.py"
        checksum_cmd = f"python3 {checksum_script} {models_path}"
        run_ssh_command(remote_host, checksum_cmd, dry_run=dry_run)

        print("✅ Checksum comparison completed")
        print()

    print_separator()
    print("✅ Remote update complete!" + (" (DRY RUN)" if dry_run else ""))
    print_separator()


def is_remote_host(arg: str) -> bool:
    """Check if an argument looks like a remote host (user@hostname)."""
    return bool(re.match(r"^[a-zA-Z0-9_-]+@[a-zA-Z0-9._-]+$", arg))


def main() -> None:
    """Main entry point."""
    # Check if first positional arg is a remote host
    remote_host = None
    args_to_parse = sys.argv[1:]

    if args_to_parse and is_remote_host(args_to_parse[0]):
        remote_host = args_to_parse[0]
        args_to_parse = args_to_parse[1:]

    parser = argparse.ArgumentParser(
        description="NAS Model Update Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs operations on NAS model storage:
  1. sync     - Syncs and verifies models from Cloudflare R2 storage
  2. format   - Updates repo (git pull) and runs TensorDataFormatter via bazel
  3. checksum - Compares checksums to verify integrity
  4. all      - Runs all operations in sequence (default)

Examples:
  # Run all operations locally
  python update_nas_model.py -p /zfs/data/official-models-ckpt-tensordata/ -u /root/utils/ \\
    --account-id YOUR_ID --access-key YOUR_KEY --secret-key YOUR_SECRET

  # Run all operations on remote server
  python update_nas_model.py root@nas-server -p /zfs/data/official-models-ckpt-tensordata/ -u /root/utils/ \\
    --account-id YOUR_ID --access-key YOUR_KEY --secret-key YOUR_SECRET

  # Only sync from R2
  python update_nas_model.py -p /data/models/ -u /root/utils/ -m sync \\
    --account-id YOUR_ID --access-key YOUR_KEY --secret-key YOUR_SECRET

  # Only format (no R2 credentials needed)
  python update_nas_model.py -p /data/models/ -u /root/utils/ -m format

  # Only verify checksums (no R2 credentials needed)
  python update_nas_model.py -p /data/models/ -u /root/utils/ -m checksum

  # Dry run - see what would be executed without making changes
  python update_nas_model.py -p /data/models/ -u /root/utils/ -m all --dry-run \\
    --account-id YOUR_ID --access-key YOUR_KEY --secret-key YOUR_SECRET
""",
    )

    parser.add_argument(
        "-p", "--path",
        required=True,
        help="Path to models directory (e.g., /zfs/data/official-models-ckpt-tensordata/)",
    )
    parser.add_argument(
        "-u", "--utils",
        required=True,
        help="Path to utils directory (e.g., /root/utils/)",
    )
    parser.add_argument(
        "-r", "--repo-path",
        help="Path to draw-things repository (default: {utils}/draw-things)",
    )
    parser.add_argument(
        "-m", "--mode",
        default="all",
        choices=["all", "sync", "format", "checksum"],
        help="Operation mode: sync, format, checksum, or all (default: all)",
    )
    parser.add_argument(
        "--account-id",
        help="Cloudflare R2 account ID (required for sync mode)",
    )
    parser.add_argument(
        "--access-key",
        help="R2 access key (required for sync mode)",
    )
    parser.add_argument(
        "--secret-key",
        help="R2 secret key (required for sync mode)",
    )
    parser.add_argument(
        "--exclude",
        help="Skip files whose name ends with this suffix (default: i8x.ckpt)",
        default="i8x.ckpt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making any changes",
    )

    args = parser.parse_args(args_to_parse)

    # Validate R2 credentials if sync mode is enabled
    if args.mode in ("all", "sync"):
        if not all([args.account_id, args.access_key, args.secret_key]):
            print(f"Error: R2 credentials are required for mode '{args.mode}'")
            print("Please provide --account-id, --access-key, and --secret-key")
            sys.exit(1)

    # Set default repo_path if not provided
    repo_path = args.repo_path if args.repo_path else os.path.join(args.utils, "draw-things")

    # Execute locally or remotely
    if remote_host:
        run_remote(
            remote_host=remote_host,
            models_path=args.path,
            utils_path=args.utils,
            repo_path=repo_path,
            mode=args.mode,
            account_id=args.account_id,
            access_key=args.access_key,
            secret_key=args.secret_key,
            dry_run=args.dry_run,
            exclude=args.exclude,
        )
    else:
        run_local(
            models_path=args.path,
            utils_path=args.utils,
            repo_path=repo_path,
            mode=args.mode,
            account_id=args.account_id,
            access_key=args.access_key,
            secret_key=args.secret_key,
            dry_run=args.dry_run,
            exclude=args.exclude,
        )


if __name__ == "__main__":
    main()
