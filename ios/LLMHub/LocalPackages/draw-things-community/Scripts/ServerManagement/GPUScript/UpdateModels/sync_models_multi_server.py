#!/usr/bin/env python3
"""
Multi-Server Model Sync with NAS HTTP Server

This script automatically manages the NAS HTTP server lifecycle - starting it
at the beginning and stopping it at the end.

Workflow:
1. Load GPU servers from gpu_servers.csv
2. Verify non-interactive SSH access to every GPU server
3. Start NAS HTTP server (if not already running)
4. Check if NAS HTTP server is accessible
5. Download NAS sha256-list.csv as source of truth
6. For each GPU server:
   a. Force refresh L1 (filesize) on GPU server (unless --skip-l1-refresh)
   b. Update checksums on GPU server using compare_checksums.py
   c. Compare GPU server CSV with NAS CSV at all levels (L1, L2, L3)
   d. Download missing/corrupted files from NAS HTTP server
      - Downloads to models_path_1 (fast NVMe)
      - If free space < 100G, moves file to models_path_2 (overflow)
      - Updates sha256sum in CSV immediately after each file download
   e. Run 'compare_checksums.py all' to fill in L1/L2 checksums for downloaded files
7. Stop NAS HTTP server

Usage:
  # Perform actual sync (sequential)
  python3 sync_models_multi_server.py

  # Parallel sync (all servers at once)
  python3 sync_models_multi_server.py --parallel

  # Dry-run mode (show what would be done without making changes)
  python3 sync_models_multi_server.py --dry-run

  # Parallel dry-run
  python3 sync_models_multi_server.py --parallel --dry-run

  # Skip L1 (filesize) refresh for faster sync
  python3 sync_models_multi_server.py --skip-l1-refresh

Notes:
  - NAS HTTP server is automatically started/stopped by this script
  - SSH access is checked sequentially before NAS startup or parallel sync
  - New host keys are accepted; changed host keys are rejected
  - Passwordless SSH authentication is required
  - Each GPU server must already have models_path_1/sha256-list.csv
  - Missing or inaccessible checksum lists fail that server without a full sync
  - In parallel mode, detailed output goes to logs/sync-{hostname}.log files
  - Terminal shows real-time progress updates every 30 seconds
  - wget uses dot format (--progress=dot:mega) for cleaner logs
  - By default, L1 (filesize) is force-refreshed on each GPU server before comparison
  - Use --skip-l1-refresh to use cached filesize values (faster but may miss changes)
  - Downloads go to models_path_1 first (fast). If free space < 100G, files are moved to models_path_2
"""

import os
import sys
import subprocess
import argparse
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
NAS_HOST = "root@dt-thpc-nas01"
NAS_IP = "64.71.166.2"
NAS_PATH = "/zfs/data/official-models-ckpt-tensordata"
HTTP_PORT = 61767
SCRIPT_DIR = Path(__file__).parent

# Global flags
DRY_RUN = False
SKIP_L1_REFRESH = False

# SSH preflight configuration
SSH_CONNECT_TIMEOUT_SECONDS = 10
SSH_COMMAND_TIMEOUT_SECONDS = 20

# Global progress tracking for parallel mode
PROGRESS_LOCK = threading.Lock()
PROGRESS_DATA = {}  # {server: {"phase": str, "files_synced": int, "total_files": int, "status": str}}


def print_step(step, total, message):
    """Print step header"""
    print(f"\n{'='*70}")
    print(f"[Step {step}/{total}] {message}")
    print('='*70)


def update_progress(server, phase, files_synced=0, total_files=0, status="running", error_msg=None):
    """Update progress data for a server (thread-safe)

    Args:
        server: Server identifier
        phase: Current phase (e.g., "Checking checksums", "Downloading", "Verifying")
        files_synced: Number of files synced so far
        total_files: Total number of files to sync
        status: Status ("running", "completed", "failed")
        error_msg: Optional error message when status is "failed"
    """
    with PROGRESS_LOCK:
        PROGRESS_DATA[server] = {
            "phase": phase,
            "files_synced": files_synced,
            "total_files": total_files,
            "status": status,
            "error_msg": error_msg
        }


def log_print(log_file, message, also_print=False):
    """Write message to log file and optionally to stdout

    Args:
        log_file: File object to write to
        message: Message to write
        also_print: If True, also print to stdout
    """
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()
    if also_print or not log_file:
        print(message)


def display_progress_status():
    """Display current progress status for all servers (thread-safe)"""
    with PROGRESS_LOCK:
        if not PROGRESS_DATA:
            return

        print("\n" + "="*70)
        print("Current Status:")
        print("="*70)

        for server, data in sorted(PROGRESS_DATA.items()):
            phase = data["phase"]
            files_synced = data["files_synced"]
            total_files = data["total_files"]
            status = data["status"]
            error_msg = data.get("error_msg")

            # Format status line
            if status == "completed":
                status_icon = "✅"
                if total_files > 0:
                    status_text = f"{status_icon} Completed ({files_synced}/{total_files} files synced)"
                else:
                    status_text = f"{status_icon} Completed (already in sync)"
            elif status == "failed":
                status_icon = "❌"
                if error_msg:
                    # Truncate error message if too long
                    short_error = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                    status_text = f"{status_icon} FAILED: {short_error}"
                else:
                    status_text = f"{status_icon} FAILED"
            else:
                if total_files > 0:
                    status_text = f"{phase}... ({files_synced}/{total_files} files)"
                else:
                    status_text = f"{phase}..."

            # Get log filename
            hostname = server.split('@')[-1].split(':')[0]
            log_file_name = f"logs/sync-{hostname}.log"

            print(f"[{hostname:15}] {status_text:40} | Log: {log_file_name}")

        print("="*70)


def load_gpu_servers(filepath="gpu_servers.csv"):
    """Load GPU servers from CSV file

    Format: remote_host, models_path_1, models_path_2 [, nas_url]
    Example: root@dfw-026-001, /mnt/models, /mnt/loraModels/models_extra
             root@dt-thpc-001, /mnt/models/official-models, /mnt/models/official-models, 192.168.88.14:8000

    Downloads go to models_path_1 (fast). If free space < 100G, files are moved to models_path_2.

    Lines starting with # are treated as comments.

    Returns:
        list of tuples: [(server_with_path1, path2, nas_url), ...]
        where server_with_path1 is "user@host:/path1", path2 is the overflow path,
        and nas_url is None for default NAS or "ip:port" for custom
    """
    import csv

    servers = []

    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found")
        sys.exit(1)

    with open(filepath, 'r', newline='') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse CSV row
            reader = csv.reader([line])
            row = next(reader)
            row = [col.strip() for col in row]

            if len(row) < 3:
                print(f"⚠️  Line {line_num}: Invalid format (expected remote_host, models_path_1, models_path_2) - {line}")
                continue

            remote_host = row[0]
            models_path_1 = row[1]
            models_path_2 = row[2]
            custom_nas_url = row[3] if len(row) >= 4 else None

            # Validate format: user@hostname
            if '@' not in remote_host:
                print(f"⚠️  Line {line_num}: Invalid format (expected user@hostname) - {remote_host}")
                continue

            # Validate models_path_1 starts with /
            if not models_path_1.startswith('/'):
                print(f"⚠️  Line {line_num}: Invalid path (expected absolute path) - {models_path_1}")
                continue

            # Validate models_path_2 starts with /
            if not models_path_2.startswith('/'):
                print(f"⚠️  Line {line_num}: Invalid path (expected absolute path) - {models_path_2}")
                continue

            server_with_path = f"{remote_host}:{models_path_1}"
            servers.append((server_with_path, models_path_2, custom_nas_url))

    print(f"✅ Loaded {len(servers)} GPU server(s)")

    # Show server configurations
    for server, path2, nas_url in servers:
        remote_host, path1 = server.rsplit(':', 1)
        hostname = remote_host.split('@')[-1]
        if path1 == path2:
            path_info = f"{path1}"
        else:
            path_info = f"{path1} -> {path2}"
        if nas_url:
            print(f"   {hostname}: {path_info} (NAS: http://{nas_url})")
        else:
            print(f"   {hostname}: {path_info}")

    return servers


def check_gpu_server_ssh_access(servers):
    """Verify non-interactive SSH access to every configured GPU server.

    Hosts are checked sequentially so first-time host-key enrollment cannot
    produce overlapping prompts when the actual sync runs in parallel.
    StrictHostKeyChecking=accept-new adds previously unseen keys while still
    rejecting changed keys. BatchMode verifies that later sync commands will
    not depend on password or passphrase prompts.

    Args:
        servers: GPU server tuples returned by load_gpu_servers().

    Returns:
        bool: True if every unique remote host is accessible, False otherwise.
    """
    remote_hosts = []
    seen_hosts = set()
    for server, _, _ in servers:
        remote_host, _ = server.rsplit(':', 1)
        if remote_host not in seen_hosts:
            seen_hosts.add(remote_host)
            remote_hosts.append(remote_host)

    print("\n🔐 Checking non-interactive SSH access to GPU servers...")
    print("   New host keys will be added to ~/.ssh/known_hosts; changed keys will be rejected.")

    failures = []
    for index, remote_host in enumerate(remote_hosts, 1):
        print(f"   [{index}/{len(remote_hosts)}] {remote_host}... ", end='', flush=True)
        command = [
            'ssh',
            '-T',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', f'ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}',
            '-o', 'ConnectionAttempts=1',
            remote_host,
            'true'
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=SSH_COMMAND_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            error = f"timed out after {SSH_COMMAND_TIMEOUT_SECONDS} seconds"
            print(f"❌ {error}")
            failures.append((remote_host, error))
            continue

        if result.returncode == 0:
            print("✅ Access confirmed")
            continue

        stderr_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        error = stderr_lines[-1] if stderr_lines else f"ssh exited with code {result.returncode}"
        print(f"❌ {error}")
        failures.append((remote_host, error))

    if failures:
        print("\n❌ SSH access check failed:")
        for remote_host, error in failures:
            print(f"   - {remote_host}: {error}")
        print("   Fix SSH host-key or key-based authentication access before syncing.")
        return False

    print(f"\n✅ SSH access confirmed for all {len(remote_hosts)} GPU server(s)")
    return True


def update_gpu_server_checksums(gpu_server, log_file=None, refresh_l1=True):
    """Update checksums on GPU server using compare_checksums.py

    This updates an existing sha256-list.csv on the GPU server and cleans up
    any corrupted files. A missing remote CSV is a fatal error for this sync.

    In dry-run mode, it only downloads the existing CSV without modifying it.

    Args:
        gpu_server: Server spec in format user@hostname:/path
        log_file: Optional file object to write logs to
        refresh_l1: If True, force refresh L1 (filesize) before comparison
    """
    log_print(log_file, f"\n📊 Updating checksums on {gpu_server}...")

    # Extract hostname for CSV file - save in logs directory
    hostname = gpu_server.split('@')[-1].split(':')[0]
    logs_dir = SCRIPT_DIR / "logs"
    gpu_csv_path = logs_dir / f"sha256-list-{hostname}.csv"
    gpu_csv_local = str(gpu_csv_path)

    if DRY_RUN:
        log_print(log_file, f"   [DRY RUN] Downloading existing checksums (read-only)")
        # In dry-run, just download the existing CSV from the server without modifying it
        # Extract user@host and path from gpu_server (format: user@host:/path)
        user_host = gpu_server.split(':')[0]  # e.g., root@dt-thpc-001
        server_path = gpu_server.split(':')[1]  # e.g., /mnt/models/official-models
        remote_csv = f"{user_host}:{server_path}/sha256-list.csv"
        try:
            gpu_csv_path.unlink(missing_ok=True)
        except OSError as e:
            log_print(log_file, f"   ❌ Could not invalidate local checksum cache: {e}")
            return None

        result = subprocess.run(
            ['scp', remote_csv, gpu_csv_local],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            log_print(log_file, f"   ✅ Downloaded existing checksums to: {gpu_csv_local}")
        else:
            gpu_csv_path.unlink(missing_ok=True)
            log_print(log_file, f"   ❌ Required remote checksum list is missing or inaccessible: {remote_csv}")
            if result.stderr.strip():
                log_print(log_file, f"   Error: {result.stderr.strip()}")
            log_print(log_file, f"   Refusing to use a stale local cache or start a full sync")
            return None

        return gpu_csv_local

    # If refresh_l1 is True and not skipped globally, force refresh L1 (filesize)
    if refresh_l1 and not SKIP_L1_REFRESH:
        log_print(log_file, f"   🔄 Refreshing L1 (filesize) on GPU server...")
        cmd_l1 = [
            'python3',
            str(SCRIPT_DIR / 'compare_checksums.py'),
            'L1',
            '--force',
            gpu_server
        ]

        if log_file:
            result = subprocess.run(cmd_l1, text=True, stdout=log_file, stderr=log_file)
        else:
            result = subprocess.run(cmd_l1, text=True)

        if result.returncode != 0:
            log_print(log_file, f"   ❌ L1 refresh failed with code {result.returncode}")
            return None
        else:
            log_print(log_file, f"   ✅ L1 (filesize) refreshed")

    # Run standard checksum update (L3 by default, fills in missing values)
    cmd = [
        'python3',
        str(SCRIPT_DIR / 'compare_checksums.py'),
        gpu_server
    ]

    # Stream output to log file if provided, otherwise to stdout
    if log_file:
        result = subprocess.run(cmd, text=True, stdout=log_file, stderr=log_file)
    else:
        result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        log_print(log_file, f"   ❌ Checksum update failed with code {result.returncode}")
        return None

    log_print(log_file, f"   ✅ GPU server checksums updated: {gpu_csv_local}")
    return gpu_csv_local


def get_files_to_download(gpu_csv_local, nas_csv_local, log_file=None):
    """Compare GPU server CSV with NAS CSV to get list of files to download

    Uses 'all' level to compare at all levels (L1, L2, L3) for comprehensive
    detection of missing or corrupted files.

    Args:
        gpu_csv_local: Local path to GPU server CSV
        nas_csv_local: Local path to NAS CSV (source of truth)
        log_file: Optional file object to write logs to

    Returns:
        list or None: Filenames that need to be downloaded, or None if the
        GPU checksum CSV is unavailable.
    """
    log_print(log_file, f"\n📋 Comparing checksums (L3/sha256sum) to determine files to download...")
    log_print(log_file, f"   GPU server CSV: {gpu_csv_local}")
    log_print(log_file, f"   NAS CSV (source of truth): {nas_csv_local}")

    # Check if GPU CSV exists locally
    if not Path(gpu_csv_local).exists():
        log_print(log_file, f"   ❌ GPU server CSV not found: {gpu_csv_local}")
        log_print(log_file, f"   Refusing to start an implicit full sync")
        return None

    # Use L3 (sha256sum) comparison only - this checks actual file content
    # Using 'all' would also check L2 (8k_sha256sum) which may not be populated
    cmd = [
        'python3',
        str(SCRIPT_DIR / 'compare_checksums.py'),
        'L3',
        gpu_csv_local,
        nas_csv_local
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output - each line is a filename to download
    files_to_download = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            files_to_download.append(line)

    if DRY_RUN:
        log_print(log_file, f"   ✅ Found {len(files_to_download)} file(s) that would be downloaded")
    else:
        log_print(log_file, f"   ✅ Found {len(files_to_download)} file(s) to download")

    return files_to_download


def start_nas_http_server():
    """Start nginx HTTP server on NAS for model distribution

    Uses nginx for high-performance file serving with optimized settings.
    The server binds to 0.0.0.0:8000 so it's accessible from both internal
    and external networks. Clients connect via NAS_IP:HTTP_PORT (external)
    or internal_ip:8000 (internal, configured in gpu_servers.csv).

    Returns:
        bool: True if server started successfully or already running, False otherwise
    """
    nas_bind_port = 8000

    print(f"\n🚀 Starting nginx HTTP server on NAS...")
    print(f"   Host: {NAS_HOST}")
    print(f"   Path: {NAS_PATH}")
    print(f"   Port: {nas_bind_port}")

    # Note: We always start the server, even in dry-run mode, because we need
    # it to download the CSV and determine what files would be synced.

    # Check if already running
    print(f"   Checking if server is already running...")
    result = subprocess.run(
        ['ssh', '-T', NAS_HOST, f'lsof -ti:{nas_bind_port}'],
        capture_output=True,
        text=True
    )
    existing_pid = result.stdout.strip()

    if existing_pid:
        print(f"   ✅ nginx already running (PID: {existing_pid})")
        return True

    # Stop any existing nginx and start fresh with optimized config
    print(f"   Starting nginx with optimized config...")
    nginx_config = f'''
worker_processes auto;
events {{
    worker_connections 4096;
    use epoll;
}}
http {{
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    directio 1m;
    output_buffers 2 1m;
    sendfile_max_chunk 1m;
    server {{
        listen {nas_bind_port} reuseport;
        root {NAS_PATH};
        autoindex on;
    }}
}}
'''
    # Write config and start nginx
    start_cmd = f'''nginx -s stop 2>/dev/null; cat > /tmp/nginx.conf <<'NGINX_EOF'
{nginx_config}
NGINX_EOF
nginx -c /tmp/nginx.conf'''

    subprocess.run(
        ['ssh', '-T', NAS_HOST, start_cmd],
        capture_output=True
    )

    print(f"   Server starting...")

    # Wait for server to start
    time.sleep(2)

    # Verify it's listening
    print(f"   Verifying server is listening...")
    result = subprocess.run(
        ['ssh', '-T', NAS_HOST, f'lsof -ti:{nas_bind_port}'],
        capture_output=True,
        text=True
    )
    verify_pid = result.stdout.strip()

    if not verify_pid:
        print(f"   ❌ nginx not listening on port {nas_bind_port}")
        return False

    print(f"   ✅ nginx started successfully")
    print(f"   PID: {verify_pid}")
    return True


def stop_nas_http_server():
    """Stop nginx HTTP server on NAS

    Returns:
        bool: True if server stopped successfully or wasn't running, False otherwise
    """
    print(f"\n🛑 Stopping nginx on NAS...")
    print(f"   Host: {NAS_HOST}")

    # Use nginx -s stop for graceful shutdown
    print(f"   Stopping nginx...")
    subprocess.run(
        ['ssh', '-T', NAS_HOST, 'nginx -s stop 2>/dev/null || true'],
        capture_output=True,
        text=True
    )

    # Wait a moment for graceful shutdown
    time.sleep(1)

    print(f"   ✅ nginx stopped")
    return True


def check_nas_http_server():
    """Check if NAS HTTP server is accessible

    Returns:
        bool: True if server is accessible, False otherwise
    """
    print(f"\n🌐 Checking NAS HTTP server...")
    print(f"   URL: http://{NAS_IP}:{HTTP_PORT}")

    if DRY_RUN:
        print(f"   [DRY RUN] Would check: http://{NAS_IP}:{HTTP_PORT}/sha256-list.csv")
        print(f"   [DRY RUN] Assuming NAS HTTP server is accessible")
        return True

    # Try to download the CSV to verify server is accessible
    test_url = f"http://{NAS_IP}:{HTTP_PORT}/sha256-list.csv"

    result = subprocess.run(
        ['wget', '--spider', '-q', test_url],
        capture_output=True,
        timeout=10
    )

    if result.returncode == 0:
        print(f"   ✅ NAS HTTP server is accessible")
        return True
    else:
        print(f"   ❌ NAS HTTP server is NOT accessible")
        return False


def download_nas_csv():
    """Download sha256-list.csv from NAS HTTP server as source of truth

    Returns:
        str: Local path to downloaded CSV file
    """
    print(f"\n📥 Downloading NAS CSV (source of truth) via HTTP...")

    nas_csv_url = f"http://{NAS_IP}:{HTTP_PORT}/sha256-list.csv"
    logs_dir = SCRIPT_DIR / "logs"
    nas_csv_local = logs_dir / 'nas-sha256-list.csv'

    if DRY_RUN:
        # In dry-run mode, still download the CSV for comparison purposes
        # This allows us to show what files would be synced
        print(f"   [DRY RUN] Downloading NAS CSV for comparison (read-only operation)")
        # Fall through to actual download logic below

    result = subprocess.run(
        ['wget', '-q', '-O', str(nas_csv_local), nas_csv_url],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        print(f"   ❌ Failed to download NAS CSV")
        if result.stderr:
            print(f"   Error: {result.stderr}")
        if DRY_RUN:
            print(f"   [DRY RUN] Cannot proceed without NAS CSV for comparison")
        return None

    if DRY_RUN:
        print(f"   ✅ Downloaded to: {nas_csv_local} (for comparison only)")
    else:
        print(f"   ✅ Downloaded to: {nas_csv_local}")
    return str(nas_csv_local)




def load_nas_checksums(nas_csv_local):
    """Load checksums from NAS CSV file

    Args:
        nas_csv_local: Path to NAS CSV file

    Returns:
        dict: {filename: sha256sum}
    """
    import csv
    checksums = {}
    with open(nas_csv_local, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            if not row:
                continue
            filename = (row.get('filename', '') or '').strip()
            sha256 = (row.get('sha256sum', '') or '').strip()
            if filename and sha256:
                checksums[filename] = sha256
    return checksums


def compute_remote_hash(hostname, filepath, log_file=None):
    """Compute SHA256 hash of a file on remote server

    Args:
        hostname: SSH hostname (user@host)
        filepath: Full path to file on remote server
        log_file: Optional file object to write logs to

    Returns:
        str: SHA256 hash or None if failed
    """
    cmd = f'sha256sum "{filepath}" | cut -d" " -f1'
    result = subprocess.run(
        ['ssh', hostname, cmd],
        capture_output=True,
        text=True,
        timeout=600  # 10 minutes for hash computation
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        log_print(log_file, f"      Failed to compute hash: {result.stderr}")
        return None


def get_remote_free_space_gb(hostname, path, log_file=None):
    """Get free space in GB on remote server path

    Args:
        hostname: SSH hostname (user@host)
        path: Path to check
        log_file: Optional file object to write logs to

    Returns:
        float: Free space in GB, or -1 if failed
    """
    cmd = f'df -BG "{path}" | tail -1 | awk \'{{print $4}}\' | tr -d G'
    result = subprocess.run(
        ['ssh', hostname, cmd],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            log_print(log_file, f"      ⚠️  Could not parse free space: {result.stdout.strip()}")
            return -1
    else:
        log_print(log_file, f"      ⚠️  Could not get free space: {result.stderr}")
        return -1


def move_remote_file(hostname, src_path, dest_path, log_file=None):
    """Move a file on remote server

    Args:
        hostname: SSH hostname (user@host)
        src_path: Source file path
        dest_path: Destination file path
        log_file: Optional file object to write logs to

    Returns:
        bool: True if move successful, False otherwise
    """
    # Ensure destination directory exists
    dest_dir = os.path.dirname(dest_path)
    cmd = f'mkdir -p "{dest_dir}" && mv "{src_path}" "{dest_path}"'
    result = subprocess.run(
        ['ssh', hostname, cmd],
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes for move (large files)
    )
    if result.returncode == 0:
        return True
    else:
        log_print(log_file, f"      ⚠️  Move failed: {result.stderr}")
        return False


# Minimum free space threshold (in GB) before moving files to overflow path
MIN_FREE_SPACE_GB = 100


def download_files_from_nas(gpu_server, files, log_file=None, server_name=None, custom_nas_url=None, nas_csv_local=None, overflow_path=None):
    """Download files from NAS HTTP server to GPU server one by one

    Uses wget -O to download and overwrite existing files automatically.
    Uses dot format (--progress=dot:mega) for cleaner log output.
    After each successful download, verifies hash against NAS CSV.
    If hash mismatch: recompute once, then retry download once.
    If still mismatch after retry, stops sync for this server.

    After download, if free space on path1 < 100G, moves file to overflow_path.

    Verification flow for each file:
        Download -> Compute Hash -> Match?
          |-- Yes -> Success, next file
          |-- No  -> Recompute Hash -> Match?
                       |-- Yes -> Success (was transient)
                       |-- No  -> Retry Download -> Compute Hash -> Match?
                                    |-- Yes -> Success
                                    |-- No  -> FATAL: Stop sync for this server

    Args:
        gpu_server: Server spec in format user@hostname:/path
        files: List of filenames to download
        log_file: Optional file object to write logs to
        server_name: Server name for progress tracking
        custom_nas_url: Optional custom NAS URL in format "ip:port"
        nas_csv_local: Path to NAS CSV file for hash verification
        overflow_path: Path to move files to when path1 is nearly full (optional)

    Returns:
        tuple: (success_count, error_message) - error_message is None if all OK,
               or a string describing the fatal error that should stop sync
    """
    if not files:
        log_print(log_file, "   ✅ No files to download")
        return (0, None)

    # Extract hostname and path
    hostname, path = gpu_server.split(':')

    # Use custom NAS URL if provided, otherwise use default
    if custom_nas_url:
        http_url = f"http://{custom_nas_url}"
    else:
        http_url = f"http://{NAS_IP}:{HTTP_PORT}"

    # Load expected checksums from NAS CSV
    expected_checksums = {}
    if nas_csv_local:
        expected_checksums = load_nas_checksums(nas_csv_local)
        log_print(log_file, f"   Loaded {len(expected_checksums)} checksums from NAS CSV")

    log_print(log_file, f"\n📥 Downloading {len(files)} file(s) from NAS...")
    log_print(log_file, f"   From: {http_url}")
    log_print(log_file, f"   To: {gpu_server}")

    if DRY_RUN:
        log_print(log_file, f"\n   [DRY RUN] Would download the following files:")
        for i, filename in enumerate(files, 1):
            file_url = f"{http_url}/{filename}"
            dest_path = f"{path}/{filename}"
            log_print(log_file, f"   [{i}/{len(files)}] {filename}")
            log_print(log_file, f"      From: {file_url}")
            log_print(log_file, f"      To:   {dest_path}")
            log_print(log_file, f"      [DRY RUN] Would verify hash after download")
        log_print(log_file, f"\n   [DRY RUN] Would download {len(files)} file(s)")
        return (len(files), None)

    success_count = 0

    for i, filename in enumerate(files, 1):
        file_url = f"{http_url}/{filename}"
        dest_path = f"{path}/{filename}"

        # Update progress
        if server_name:
            update_progress(server_name, "Downloading", i-1, len(files))

        # Show progress
        log_print(log_file, f"   [{i}/{len(files)}] {filename}")

        # Get expected hash
        expected_hash = expected_checksums.get(filename)
        if not expected_hash:
            log_print(log_file, f"      ⚠️  No expected hash found in NAS CSV, skipping verification")

        # Try download (with one retry on hash mismatch)
        max_attempts = 2
        download_success = False
        last_failure_reason = None  # Track why the last attempt failed

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                log_print(log_file, f"      🔄 Retry attempt {attempt}/{max_attempts}...")

            # Download file using wget with dot format for cleaner logs
            # -O will overwrite existing file automatically (no need for rm)
            # --progress=dot:mega shows progress in dot format (cleaner for logs)
            wget_cmd = f'wget --progress=dot:mega "{file_url}" -O "{dest_path}"'

            # Stream output to log file if provided
            if log_file:
                result = subprocess.run(
                    ['ssh', hostname, wget_cmd],
                    text=True,
                    stdout=log_file,
                    stderr=log_file,
                    timeout=3600  # 1 hour per file
                )
            else:
                result = subprocess.run(
                    ['ssh', hostname, wget_cmd],
                    text=True,
                    timeout=3600  # 1 hour per file
                )

            if result.returncode != 0:
                log_print(log_file, f"      ❌ wget failed with code {result.returncode}")
                last_failure_reason = f"wget failed with code {result.returncode}"
                continue  # Try again if we have retries left

            # Download succeeded, now verify hash
            if not expected_hash:
                # No expected hash, skip verification
                download_success = True
                break

            # Compute hash of downloaded file
            log_print(log_file, f"      Computing hash...")
            computed_hash = compute_remote_hash(hostname, dest_path, log_file)

            if computed_hash is None:
                log_print(log_file, f"      ❌ Failed to compute hash")
                last_failure_reason = "hash computation failed"
                continue  # Try again

            if computed_hash == expected_hash:
                log_print(log_file, f"      ✅ Hash verified: {computed_hash[:16]}...")
                download_success = True
                break

            # Hash mismatch - recompute once to rule out transient error
            log_print(log_file, f"      ⚠️  Hash mismatch! Expected: {expected_hash[:16]}..., Got: {computed_hash[:16]}...")
            log_print(log_file, f"      Recomputing hash to confirm...")

            recomputed_hash = compute_remote_hash(hostname, dest_path, log_file)

            if recomputed_hash == expected_hash:
                log_print(log_file, f"      ✅ Hash verified on recompute: {recomputed_hash[:16]}...")
                download_success = True
                break

            # Still mismatch after recompute
            if recomputed_hash is None:
                log_print(log_file, f"      ❌ Recompute hash failed")
                last_failure_reason = "hash recomputation failed"
            elif recomputed_hash != computed_hash:
                log_print(log_file, f"      ⚠️  Recomputed hash differs: {recomputed_hash[:16]}...")
                last_failure_reason = f"hash mismatch (expected: {expected_hash[:16]}..., got: {recomputed_hash[:16]}...)"
            else:
                last_failure_reason = f"hash mismatch (expected: {expected_hash[:16]}..., got: {computed_hash[:16]}...)"

            log_print(log_file, f"      ❌ Hash verification failed after recompute")
            # Will retry download if attempts remain

        if download_success:
            success_count += 1

            # Update sha256-list.csv on the GPU server with verified hash
            # Uses 4-column format: filename,sha256sum,8k_sha256sum,filesize
            log_print(log_file, f"      Updating checksum in CSV...")
            checksum_to_save = expected_hash if expected_hash else computed_hash
            if checksum_to_save:
                # Filter out both old (2-col) and new (4-col) headers, preserve existing data
                checksum_cmd = f'cd "{path}" && ' \
                              f'{{ echo "filename,sha256sum,8k_sha256sum,filesize"; ' \
                              f'{{ cat sha256-list.csv 2>/dev/null | grep -v "^filename," | grep -v "^{filename}," || true; }} ; ' \
                              f'echo "{filename},{checksum_to_save},,"; }} | ' \
                              f'{{ read header; echo "$header"; sort -t, -k1; }} > /tmp/sorted.csv && mv /tmp/sorted.csv sha256-list.csv'

                checksum_result = subprocess.run(
                    ['ssh', hostname, checksum_cmd],
                    text=True,
                    capture_output=True,
                    timeout=60
                )

                if checksum_result.returncode == 0:
                    log_print(log_file, f"      ✅ Checksum saved to CSV")
                else:
                    log_print(log_file, f"      ⚠️  Failed to update CSV: {checksum_result.stderr}")
            else:
                log_print(log_file, f"      ⚠️  No checksum to save")

            # Check free space and move to overflow path if needed
            if overflow_path and overflow_path != path:
                free_space_gb = get_remote_free_space_gb(hostname, path, log_file)
                if free_space_gb >= 0 and free_space_gb < MIN_FREE_SPACE_GB:
                    log_print(log_file, f"      📦 Free space {free_space_gb:.0f}G < {MIN_FREE_SPACE_GB}G, moving to overflow path...")
                    src_file = dest_path
                    dest_file = f"{overflow_path}/{filename}"
                    if move_remote_file(hostname, src_file, dest_file, log_file):
                        log_print(log_file, f"      ✅ Moved to {overflow_path}")
                    else:
                        log_print(log_file, f"      ⚠️  Failed to move file, keeping in {path}")

            # Update progress after successful download
            if server_name:
                update_progress(server_name, "Downloading", success_count, len(files))
        else:
            # Failed after all retries - this is a fatal error, stop sync for this server
            error_msg = f"{filename}: {last_failure_reason}"
            log_print(log_file, f"   ❌ FATAL: {error_msg}")
            log_print(log_file, f"   ❌ Stopping sync for this server")
            return (success_count, error_msg)

    log_print(log_file, f"   ✅ Downloaded and verified {success_count}/{len(files)} file(s)")
    return (success_count, None)




def sync_single_server(gpu_server, nas_csv_local, log_file=None, server_name=None, custom_nas_url=None, overflow_path=None):
    """Sync a single GPU server with NAS

    Workflow:
    1. Update checksums on GPU server
    2. Compare checksums to get files to download
    3. Download files from NAS HTTP server (updates sha256sum in CSV after each file)
       - If free space < 100G, moves files to overflow_path
    4. Run 'compare_checksums.py all' to fill in L1/L2 checksums for downloaded files

    Args:
        gpu_server: Server spec in format user@hostname:/path
        nas_csv_local: Local path to NAS CSV file (source of truth)
        log_file: Optional file object to write logs to
        server_name: Server name for progress tracking
        custom_nas_url: Optional custom NAS URL in format "ip:port"
        overflow_path: Path to move files when primary path is nearly full

    Returns:
        bool: True if sync successful, False otherwise
    """
    log_print(log_file, f"\n{'='*70}")
    log_print(log_file, f"Syncing: {gpu_server}")
    if custom_nas_url:
        log_print(log_file, f"Using custom NAS: http://{custom_nas_url}")
    if overflow_path:
        _, path = gpu_server.split(':')
        if overflow_path != path:
            log_print(log_file, f"Overflow path: {overflow_path}")
    log_print(log_file, '='*70)

    try:
        # Step 1: Update checksums on GPU server
        if server_name:
            update_progress(server_name, "Checking checksums", 0, 0)
        if not log_file:
            print_step(1, 4, f"Update checksums on {gpu_server}")
        else:
            log_print(log_file, f"\n[Step 1/4] Update checksums on {gpu_server}")
        gpu_csv_local = update_gpu_server_checksums(gpu_server, log_file)
        if gpu_csv_local is None:
            error_msg = f"Required remote checksum list is missing or inaccessible: {gpu_server}/sha256-list.csv"
            if server_name:
                update_progress(server_name, "Failed", 0, 0, status="failed", error_msg=error_msg)
            log_print(log_file, f"\n❌ {error_msg}")
            log_print(log_file, "❌ Stopping sync for this server")
            return False

        # Step 2: Compare checksums to get files to download
        if server_name:
            update_progress(server_name, "Comparing checksums", 0, 0)
        if not log_file:
            print_step(2, 4, "Compare checksums")
        else:
            log_print(log_file, f"\n[Step 2/4] Compare checksums")
        files_to_download = get_files_to_download(gpu_csv_local, nas_csv_local, log_file)

        if files_to_download is None:
            error_msg = f"Local checksum list is unavailable for {gpu_server}"
            if server_name:
                update_progress(server_name, "Failed", 0, 0, status="failed", error_msg=error_msg)
            log_print(log_file, f"\n❌ {error_msg}")
            log_print(log_file, "❌ Stopping sync for this server")
            return False

        if not files_to_download:
            if server_name:
                update_progress(server_name, "Completed", 0, 0, status="completed")
            log_print(log_file, "\n✅ Server already in sync - no files to download")
            return True

        # Step 3: Download files from NAS
        if server_name:
            update_progress(server_name, "Downloading", 0, len(files_to_download))
        if not log_file:
            print_step(3, 4, "Download files from NAS")
        else:
            log_print(log_file, f"\n[Step 3/4] Download files from NAS")
        success_count, error_msg = download_files_from_nas(
            gpu_server, files_to_download, log_file, server_name, custom_nas_url, nas_csv_local, overflow_path
        )

        # Check for fatal error (hash verification failure)
        if error_msg:
            if server_name:
                update_progress(server_name, "Failed", success_count, len(files_to_download), status="failed", error_msg=error_msg)
            log_print(log_file, f"\n❌ Sync stopped due to error: {error_msg}")
            return False

        if success_count < len(files_to_download):
            log_print(log_file, f"\n⚠️  Warning: Only {success_count}/{len(files_to_download)} files downloaded successfully")

        # Update all checksums (L1, L2, L3) for downloaded files using 'all' mode
        # This fills in missing values only - L3 (sha256sum) is already saved during download
        if success_count > 0 and not DRY_RUN:
            if server_name:
                update_progress(server_name, "Updating checksums", success_count, len(files_to_download))
            log_print(log_file, f"\n📊 Updating checksums for downloaded files (all levels)...")
            cmd = [
                'python3',
                str(SCRIPT_DIR / 'compare_checksums.py'),
                'all',
                gpu_server
            ]
            if log_file:
                result = subprocess.run(cmd, text=True, stdout=log_file, stderr=log_file)
            else:
                result = subprocess.run(cmd, text=True)
            if result.returncode != 0:
                error_msg = f"Final checksum update failed with code {result.returncode}"
                if server_name:
                    update_progress(
                        server_name,
                        "Failed",
                        success_count,
                        len(files_to_download),
                        status="failed",
                        error_msg=error_msg
                    )
                log_print(log_file, f"   ❌ {error_msg}")
                return False
            else:
                log_print(log_file, f"   ✅ Checksums updated")

        if server_name:
            update_progress(server_name, "Completed", success_count, len(files_to_download), status="completed")
        log_print(log_file, f"\n✅ Server sync completed: {success_count}/{len(files_to_download)} files downloaded")
        return True

    except Exception as e:
        if server_name:
            update_progress(server_name, "Failed", 0, 0, status="failed")
        log_print(log_file, f"\n❌ Error syncing {gpu_server}: {e}")
        import traceback
        if log_file:
            traceback.print_exc(file=log_file)
        else:
            traceback.print_exc()
        return False


def sync_server_with_logging(server, nas_csv_local, log_dir=".", custom_nas_url=None, overflow_path=None):
    """Wrapper to sync a single server with logging to file

    Args:
        server: Server spec in format user@hostname:/path
        nas_csv_local: Local path to NAS CSV file (source of truth)
        log_dir: Directory to write log files to
        custom_nas_url: Optional custom NAS URL in format "ip:port"
        overflow_path: Path to move files when primary path is nearly full

    Returns:
        bool: True if sync successful, False otherwise
    """
    # Extract hostname for log file naming
    hostname = server.split('@')[-1].split(':')[0]
    log_filename = f"{log_dir}/sync-{hostname}.log"

    # Open log file
    with open(log_filename, 'w') as log_file:
        # Write header to log
        log_file.write("="*70 + "\n")
        log_file.write(f"Sync Log for {server}\n")
        if custom_nas_url:
            log_file.write(f"Custom NAS: http://{custom_nas_url}\n")
        if overflow_path:
            _, path = server.split(':')
            if overflow_path != path:
                log_file.write(f"Overflow path: {overflow_path}\n")
        log_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*70 + "\n\n")
        log_file.flush()

        # Sync the server
        success = sync_single_server(server, nas_csv_local, log_file, hostname, custom_nas_url, overflow_path)

        # Write footer to log
        log_file.write("\n" + "="*70 + "\n")
        log_file.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
        log_file.write("="*70 + "\n")

    return success


def main():
    global DRY_RUN, SKIP_L1_REFRESH

    nas_http_server_start_attempted = False

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Multi-Server Model Sync with NAS HTTP Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perform actual sync (sequential)
  python3 sync_models_multi_server.py

  # Parallel sync (all servers at once)
  python3 sync_models_multi_server.py --parallel

  # Dry-run mode (show what would be done without making changes)
  python3 sync_models_multi_server.py --dry-run

  # Parallel dry-run
  python3 sync_models_multi_server.py --parallel --dry-run

  # Skip L1 (filesize) refresh for faster sync (use cached values)
  python3 sync_models_multi_server.py --skip-l1-refresh
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Sync all servers in parallel (faster but more resource intensive)'
    )
    parser.add_argument(
        '--skip-l1-refresh',
        action='store_true',
        help='Skip L1 (filesize) refresh before comparison (use cached values)'
    )

    args = parser.parse_args()

    DRY_RUN = args.dry_run
    SKIP_L1_REFRESH = args.skip_l1_refresh

    # Create logs directory if it doesn't exist
    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Multi-Server Model Sync with NAS HTTP Server")
    if DRY_RUN:
        print("[DRY RUN MODE - No changes will be made]")
    if args.parallel:
        print("[PARALLEL MODE - All servers synced simultaneously]")
    if SKIP_L1_REFRESH:
        print("[SKIP L1 REFRESH - Using cached filesize values]")
    else:
        print("[L1 REFRESH ENABLED - Refreshing filesize on each GPU server]")
    print("="*70)
    print(f"NAS Source: {NAS_HOST}:{NAS_PATH}")
    print(f"NAS HTTP: http://{NAS_IP}:{HTTP_PORT}")
    print(f"Server List: gpu_servers.csv")

    try:
        # Step 1: Load GPU servers
        print_step(1, 6, "Load GPU servers from gpu_servers.csv")
        servers = load_gpu_servers()

        if not servers:
            print("❌ No servers found in gpu_servers.csv")
            sys.exit(1)

        print(f"\nServers to sync:")
        for i, (server, path2, custom_nas) in enumerate(servers, 1):
            _, path1 = server.rsplit(':', 1)
            if path1 == path2:
                path_info = path1
            else:
                path_info = f"{path1} -> {path2}"
            if custom_nas:
                print(f"  {i}. {server.split(':')[0]}: {path_info} (NAS: http://{custom_nas})")
            else:
                print(f"  {i}. {server.split(':')[0]}: {path_info}")

        # Step 2: Verify all GPU servers are ready for non-interactive SSH
        print_step(2, 6, "Check SSH access to GPU servers")
        if not check_gpu_server_ssh_access(servers):
            print("\n❌ Not all GPU servers are accessible. Exiting before sync.")
            sys.exit(1)

        # Step 3: Start NAS HTTP server
        print_step(3, 6, "Start NAS HTTP server")
        nas_http_server_start_attempted = True
        if not start_nas_http_server():
            print("\n❌ Failed to start NAS HTTP server. Exiting.")
            sys.exit(1)

        # Step 4: Check if NAS HTTP server is accessible
        print_step(4, 6, "Check NAS HTTP server")
        if not check_nas_http_server():
            print("\n❌ NAS HTTP server is not accessible. Exiting.")
            sys.exit(1)

        # Step 5: Download NAS CSV (source of truth)
        print_step(5, 6, "Download NAS CSV (source of truth)")
        nas_csv_local = download_nas_csv()
        if not nas_csv_local:
            print("❌ Failed to download NAS CSV")
            sys.exit(1)

        # Step 6: Sync each GPU server
        print_step(6, 6, "Sync GPU servers")

        results = {}

        if args.parallel:
            # Parallel execution - sync all servers simultaneously
            print(f"\n🚀 Syncing {len(servers)} servers in parallel...")
            print(f"   Detailed logs will be written to logs/sync-*.log files\n")

            # Display initial status
            for server, path2, custom_nas in servers:
                hostname = server.split('@')[-1].split(':')[0]
                log_filename = f"logs/sync-{hostname}.log"
                update_progress(hostname, "Starting", 0, 0)
                print(f"[{hostname:15}] Starting...{' '*25} | Log: {log_filename}")

            print()

            # Start monitoring thread
            stop_monitoring = threading.Event()

            def monitor_progress():
                """Monitor and display progress periodically"""
                while not stop_monitoring.is_set():
                    time.sleep(30)  # Update every 30 seconds
                    if not stop_monitoring.is_set():
                        display_progress_status()

            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()

            # Sync all servers in parallel
            with ThreadPoolExecutor(max_workers=len(servers)) as executor:
                # Submit all server sync tasks
                future_to_server = {
                    executor.submit(sync_server_with_logging, server, nas_csv_local, str(logs_dir), custom_nas, path2): (server, path2, custom_nas)
                    for server, path2, custom_nas in servers
                }

                # Process results as they complete
                for future in as_completed(future_to_server):
                    server, path2, custom_nas = future_to_server[future]
                    hostname = server.split('@')[-1].split(':')[0]
                    try:
                        success = future.result()
                        results[server] = success

                        # Display completion message
                        with PROGRESS_LOCK:
                            data = PROGRESS_DATA.get(hostname, {})
                            files_synced = data.get("files_synced", 0)
                            total_files = data.get("total_files", 0)
                            error_msg = data.get("error_msg")
                            if not success:
                                failure_detail = error_msg or f"See logs/sync-{hostname}.log"
                                print(f"\n[{hostname}] ❌ Failed: {failure_detail}")
                            elif total_files > 0:
                                print(f"\n[{hostname}] ✅ Completed: {files_synced}/{total_files} files synced")
                            else:
                                print(f"\n[{hostname}] ✅ Completed: Already in sync")
                    except Exception as e:
                        print(f"\n[{hostname}] ❌ Exception: {e}")
                        results[server] = False

            # Stop monitoring thread
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

            # Final status display
            display_progress_status()

        else:
            # Sequential execution - sync servers one at a time
            for i, (server, path2, custom_nas) in enumerate(servers, 1):
                print(f"\n{'#'*70}")
                print(f"# Server {i}/{len(servers)}")
                print(f"{'#'*70}")

                success = sync_single_server(server, nas_csv_local, custom_nas_url=custom_nas, overflow_path=path2)
                results[server] = success

        # Summary
        print("\n" + "="*70)
        print("Sync Summary")
        print("="*70)

        success_count = sum(1 for v in results.values() if v)

        for server, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{status}: {server}")

        print(f"\nTotal: {success_count}/{len(servers)} servers synced successfully")
        print("="*70)

        if success_count == len(servers):
            print("\n🎉 All servers synced successfully!")
            exit_code = 0
        else:
            print(f"\n⚠️  {len(servers) - success_count} server(s) failed to sync")
            exit_code = 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Sync interrupted by user")
        exit_code = 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        # Only clean up after this run attempted to start the NAS server.
        if nas_http_server_start_attempted:
            try:
                stop_nas_http_server()
            except KeyboardInterrupt:
                print("\n⚠️  NAS HTTP server cleanup interrupted")
                exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
