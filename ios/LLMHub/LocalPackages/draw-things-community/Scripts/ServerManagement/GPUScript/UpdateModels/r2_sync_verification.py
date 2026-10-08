#!/usr/bin/env python3
import os
import sys
import boto3
import argparse
import logging
import hashlib
import requests
import concurrent.futures
import time
import re
from datetime import datetime
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm
import json
import schedule
import subprocess
import shutil

class R2ModelSync:
    def __init__(self, account_id, access_key_id, secret_access_key, bucket_name, target_dir,
                 enable_cleanup=False, max_retries=3, size_tolerance_mb=10, exclude_suffix=None):
        self.account_id = account_id
        self.bucket_name = bucket_name
        self.target_dir = target_dir
        self.base_url = f"https://static.libnnc.org"
        self.enable_cleanup = enable_cleanup
        self.max_retries = max_retries
        self.size_tolerance_mb = size_tolerance_mb  # Size tolerance in MB
        self.size_tolerance_bytes = size_tolerance_mb * 1024 * 1024  # Convert to bytes
        self.exclude_suffix = exclude_suffix
        self.r2_client = self._setup_r2_client(account_id, access_key_id, secret_access_key)
        self.sha256_dict = self._get_sha256_dict()

    def _get_sha256_dict(self):
        """Download and parse SHA256 files from drawthingsai repository"""
        base_url = "https://raw.githubusercontent.com/drawthingsai/community-models/json/docs"
        files = [
            "controlnets_sha256.json",
            "embeddings_sha256.json",
            "loras_sha256.json",
            "models_sha256.json"
        ]
        
        sha256_dict = {}
        
        for file in files:
            url = f"{base_url}/{file}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse markdown-style content
                content = response.text
                sha256_dict = json.loads(content) | sha256_dict
                    
            except requests.RequestException as e:
                logging.error(f"Error downloading {file}: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
        
        logging.info(f"Loaded {len(sha256_dict)} SHA256 hashes from repository")
        return sha256_dict
        
    def _calculate_sha256(self, filepath):
        """Calculate SHA256 hash of file in chunks with progress bar"""
        sha256_hash = hashlib.sha256()
        chunk_size = 1024 * 1024  # 1MB chunks
        file_size = os.path.getsize(filepath)
        
        try:
            with open(filepath, 'rb') as f:
                with tqdm(total=file_size, 
                         unit='iB', 
                         unit_scale=True, 
                         desc=f"Calculating SHA256 for {os.path.basename(filepath)}") as pbar:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        sha256_hash.update(chunk)
                        pbar.update(len(chunk))
            return sha256_hash.hexdigest()
        except Exception as e:
            logging.error(f"Failed to calculate SHA256 for {filepath}: {str(e)}")
            return None

    def _setup_r2_client(self, account_id, access_key_id, secret_access_key):
        """Set up R2 client using Cloudflare credentials"""
        r2_endpoint = f'https://{account_id}.r2.cloudflarestorage.com'
        config = Config(
            region_name='auto',
            s3={'addressing_style': 'virtual', 'endpoint_url': r2_endpoint},
            retries={'max_attempts': self.max_retries}
        )
        return boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=config,
            verify=False
        )

    def _get_remote_objects(self):
        """Get list of objects from R2"""
        try:
            objects = {}
            paginator = self.r2_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects[obj['Key']] = {
                            'size': obj['Size'],
                            'last_modified': obj['LastModified']
                        }
            return objects
        except ClientError as e:
            logging.error(f"Failed to list R2 objects: {str(e)}")
            return {}

    def _calculate_md5(self, filepath):
        """Calculate MD5 hash of file in chunks with progress bar"""
        md5_hash = hashlib.md5()
        chunk_size = 1024 * 1024  # 1MB chunks
        file_size = os.path.getsize(filepath)
        
        try:
            with open(filepath, 'rb') as f:
                with tqdm(total=file_size, 
                         unit='iB', 
                         unit_scale=True, 
                         desc=f"Verifying {os.path.basename(filepath)}") as pbar:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        md5_hash.update(chunk)
                        pbar.update(len(chunk))
            return md5_hash.hexdigest()
        except Exception as e:
            logging.error(f"Failed to calculate MD5 for {filepath}: {str(e)}")
            return None
    
    def _check_file_size_match(self, local_path, expected_size):
        """Check if local file size matches expected size within tolerance"""
        try:
            # Check if this is a .ckpt file and if corresponding .ckpt-tensordata file exists
            if local_path.endswith('.ckpt'):
                tensordata_path = local_path + '-tensordata'
                if os.path.exists(tensordata_path):
                    # Skip size verification if tensordata file exists
                    actual_size = os.path.getsize(local_path)
                    tensordata_size = os.path.getsize(tensordata_path)
                    combined_size = actual_size + tensordata_size
                    
                    # Format sizes for display
                    def format_size(size_bytes):
                        size_mb = size_bytes / (1024 * 1024)
                        if size_mb >= 1024:
                            return f"{size_mb/1024:.2f} GB"
                        else:
                            return f"{size_mb:.2f} MB"
                    
                    expected_str = format_size(expected_size)
                    combined_str = format_size(combined_size)
                    ckpt_str = format_size(actual_size)
                    tensordata_str = format_size(tensordata_size)
                    
                    # Check if combined size difference is significant (> 30MB)
                    size_diff_bytes = abs(combined_size - expected_size)
                    size_diff_mb = size_diff_bytes / (1024 * 1024)
                    
                    logging.info(f"Skipping size verification for {os.path.basename(local_path)} - found corresponding .ckpt-tensordata file")
                    
                    if size_diff_mb > 30:
                        logging.warning(f"  ⚠️  LARGE SIZE DIFFERENCE: Combined size: {combined_str} ({ckpt_str} + {tensordata_str}) vs expected: {expected_str} (diff: {size_diff_mb:.1f} MB)")
                    else:
                        logging.info(f"  Combined size: {combined_str} ({ckpt_str} + {tensordata_str}) vs expected: {expected_str}")
                    
                    return True, actual_size, 0
            
            actual_size = os.path.getsize(local_path)
            size_diff = abs(actual_size - expected_size)
            
            if size_diff <= self.size_tolerance_bytes:
                return True, actual_size, size_diff
            else:
                return False, actual_size, size_diff
        except OSError:
            return False, 0, expected_size
    
    def _get_local_objects(self):
        """Get list of local files with size information"""
        local_objects = {}
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                if file != 'r2_sync.log' and not root.endswith('/logs'):
                    filepath = os.path.join(root, file)
                    relpath = os.path.relpath(filepath, self.target_dir)
                    try:
                        size = os.path.getsize(filepath)
                        local_objects[relpath] = {'size': size}
                    except OSError:
                        local_objects[relpath] = {'size': 0}
        return local_objects

    def _download_file(self, download_info):
        """Download a single file from R2 with retry support and resumable downloads using wget"""
        import subprocess
        import shutil
        import threading
        
        key, remote_info = download_info
        local_path = os.path.join(self.target_dir, key)
        temp_path = f"{local_path}.temp"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Check if wget is available
        if not shutil.which('wget'):
            logging.error("wget is not available on this system. Please install wget.")
            return False
        
        url = f"{self.base_url}/{key}"
        download_id = os.path.basename(key)
        
        # Create a lock for console output if it doesn't exist
        if not hasattr(self.__class__, '_console_lock'):
            self.__class__._console_lock = threading.Lock()
        
        for attempt in range(self.max_retries):
            try:
                # Check if we can resume a previous download
                if os.path.exists(temp_path):
                    current_size = os.path.getsize(temp_path)
                    if current_size >= remote_info['size']:
                        # File is already complete or larger than expected, remove and restart
                        os.remove(temp_path)
                        logging.info(f"[{download_id}] Removing oversized temp file and restarting")
                
                # Build wget command - use dot progress for concurrent downloads
                wget_cmd = [
                    'wget',
                    "--quiet",
                    '--tries=3',  # wget's internal retries
                    '--timeout=30',  # 30 second timeout
                    '--continue',  # Resume partial downloads
                    '--output-document', temp_path,  # Output to temp file
                    url
                ]
                
                logging.info(f"[{download_id}] Starting download (attempt {attempt + 1}/{self.max_retries})")
                
                # Check if resuming
                if os.path.exists(temp_path):
                    existing_size = os.path.getsize(temp_path)
                    if existing_size > 0:
                        size_mb = existing_size / (1024 * 1024)
                        if size_mb >= 1024:
                            size_str = f"{size_mb/1024:.2f} GB"
                        else:
                            size_str = f"{size_mb:.2f} MB"
                        logging.info(f"[{download_id}] Resuming from {size_str}")
                
                # Run wget with its output prefixed
                try:
                    # Start the wget process
                    process = subprocess.Popen(
                        wget_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    # Read and prefix each line of output
                    for line in process.stdout:
                        if line.strip():
                            # Add file identifier prefix to each line
                            with self.__class__._console_lock:
                                print(f"[{download_id}] {line.strip()}")
                    
                    # Wait for process to complete
                    return_code = process.wait()
                    
                    if return_code != 0:
                        raise ValueError(f"wget failed with exit code {return_code}")
                    
                    # Download completed successfully
                    with self.__class__._console_lock:
                        print(f"[{download_id}] ✓ Download completed")
                    logging.info(f"[{download_id}] Download completed")
                    
                except subprocess.CalledProcessError as e:
                    raise ValueError(f"wget failed with exit code {e.returncode}")
                
                # Verify downloaded file exists and size is reasonable
                if not os.path.exists(temp_path):
                    raise ValueError(f"Downloaded file not found: {temp_path}")
                
                downloaded_size = os.path.getsize(temp_path)
                expected_size = remote_info['size']
                size_diff = abs(downloaded_size - expected_size)
                
                # Check size within tolerance
                if size_diff > self.size_tolerance_bytes:
                    expected_mb = expected_size / (1024 * 1024)
                    actual_mb = downloaded_size / (1024 * 1024)
                    diff_mb = size_diff / (1024 * 1024)
                    
                    logging.error(f"[{download_id}] Size mismatch:")
                    logging.error(f"  Expected: {expected_mb:.2f} MB")
                    logging.error(f"  Downloaded: {actual_mb:.2f} MB")
                    logging.error(f"  Difference: {diff_mb:.2f} MB (tolerance: {self.size_tolerance_mb} MB)")
                    
                    os.remove(temp_path)
                    raise ValueError(f"Downloaded file size mismatch for {key}")
                
                # Rename temp file to final file
                os.replace(temp_path, local_path)
                
                # Format size for display
                size_mb = downloaded_size / (1024 * 1024)
                if size_mb >= 1024:
                    size_str = f"{size_mb/1024:.2f} GB"
                else:
                    size_str = f"{size_mb:.2f} MB"
                
                logging.info(f"[{download_id}] File saved ({size_str})")
                
                # Verify SHA256 if available
                if os.path.basename(key) in self.sha256_dict:
                    expected_sha256 = self.sha256_dict[os.path.basename(key)]
                    
                    with self.__class__._console_lock:
                        print(f"[{download_id}] Verifying SHA256...")
                    logging.info(f"[{download_id}] Starting SHA256 verification...")
                    
                    actual_sha256 = self._calculate_sha256(local_path)
                    
                    if actual_sha256 == expected_sha256:
                        with self.__class__._console_lock:
                            print(f"[{download_id}] ✓ SHA256 verified")
                        logging.info(f"[{download_id}] SHA256 verification successful")
                    else:
                        logging.error(f"[{download_id}] SHA256 mismatch:")
                        logging.error(f"  Expected: {expected_sha256}")
                        logging.error(f"  Got:      {actual_sha256}")
                        os.remove(local_path)
                        raise ValueError(f"SHA256 verification failed for {key}")
                else:
                    logging.warning(f"[{download_id}] No SHA256 hash found in repository for verification")
                
                with self.__class__._console_lock:
                    print(f"[{download_id}] ✓ Successfully completed ({size_str})")
                logging.info(f"[{download_id}] Successfully completed")
                return True
                
            except Exception as e:
                logging.error(f"[{download_id}] Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    # Final attempt failed, cleanup
                    with self.__class__._console_lock:
                        print(f"[{download_id}] ✗ Failed after {self.max_retries} attempts")
                    logging.error(f"[{download_id}] Failed after {self.max_retries} attempts")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False
                else:
                    # Wait before retrying with exponential backoff
                    wait_time = 2 ** attempt
                    logging.info(f"[{download_id}] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
        
        return False

    def _cleanup_temp_files(self):
        """Clean up any .temp files from previous interrupted downloads"""
        count = 0
        for root, _, files in os.walk(self.target_dir):
            for file in files:
                if file.endswith('.temp'):
                    try:
                        filepath = os.path.join(root, file)
                        os.remove(filepath)
                        count += 1
                    except OSError as e:
                        logging.error(f"Failed to remove temp file {filepath}: {str(e)}")
        if count > 0:
            logging.info(f"Cleaned up {count} temporary files")

    def _print_diff(self, local_objects, remote_objects):
        """Print files that exist locally but not in R2"""
        local_only = set(local_objects.keys()) - set(remote_objects.keys())
        if local_only:
            logging.info("\nFiles existing only locally (not in R2):")
            for file in sorted(local_only):
                try:
                    size = os.path.getsize(os.path.join(self.target_dir, file))
                    size_str = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
                    logging.info(f"  - {file} ({size_str})")
                except OSError:
                    logging.info(f"  - {file} (size unknown)")
        return local_only
    
    def sync(self, max_workers=5):
        """Synchronize local directory with R2 bucket"""
        # Clean up any temporary files first
        self._cleanup_temp_files()
        
        remote_objects = self._get_remote_objects()
        local_objects = self._get_local_objects()
        
        # Determine files to download or re-download
        to_download = []
        size_mismatches = []
        skipped_files = []

        for key, remote_info in remote_objects.items():
            if self.exclude_suffix and os.path.basename(key).endswith(self.exclude_suffix):
                skipped_files.append(key)
                continue

            local_path = os.path.join(self.target_dir, key)

            if key not in local_objects:
                # File doesn't exist locally, need to download
                to_download.append((key, remote_info))
            else:
                # File exists locally, check size
                size_match, actual_size, size_diff = self._check_file_size_match(local_path, remote_info['size'])
                
                if not size_match:
                    # Size mismatch, need to re-download
                    to_download.append((key, remote_info))
                    size_mismatches.append({
                        'key': key,
                        'expected_size': remote_info['size'],
                        'actual_size': actual_size,
                        'size_diff': size_diff
                    })
                    
                    # Log the size mismatch
                    expected_mb = remote_info['size'] / (1024 * 1024)
                    actual_mb = actual_size / (1024 * 1024)
                    diff_mb = size_diff / (1024 * 1024)
                    
                    logging.warning(f"Size mismatch for {key}:")
                    logging.warning(f"  Expected: {expected_mb:.2f} MB")
                    logging.warning(f"  Actual:   {actual_mb:.2f} MB")
                    logging.warning(f"  Diff:     {diff_mb:.2f} MB (tolerance: {self.size_tolerance_mb} MB)")
        
        # Log sync status and list files to be downloaded
        logging.info(f"Found {len(remote_objects)} files in R2")
        logging.info(f"Found {len(local_objects)} files in Local path")
        logging.info(f"Size tolerance: {self.size_tolerance_mb} MB")

        if skipped_files:
            logging.info(f"  - {len(skipped_files)} files skipped (exclude suffix: {self.exclude_suffix}):")
            for f in sorted(skipped_files):
                logging.info(f"      {f}")

        if size_mismatches:
            logging.info(f"  - {len(size_mismatches)} files with size mismatches to re-download")

        new_downloads = len(to_download) - len(size_mismatches)
        if new_downloads > 0:
            logging.info(f"  - {new_downloads} new files to download")

        unchanged_files = len(remote_objects) - len(to_download)
        logging.info(f"  - {unchanged_files} files unchanged")

        if len(local_objects) != unchanged_files:
            logging.warning(f"Found local files {len(local_objects)}, doesn't equal to {unchanged_files} files unchanged ")
            self._print_diff(local_objects, remote_objects)

        if to_download:
            logging.info("\nFiles to be downloaded:")
            for key, info in to_download:
                size_mb = info['size'] / (1024 * 1024)
                if size_mb >= 1024:  # If size is >= 1GB
                    size_str = f"{size_mb/1024:.2f} GB"
                else:
                    size_str = f"{size_mb:.2f} MB"
                
                # Check if this is a re-download due to size mismatch
                size_mismatch_info = next((sm for sm in size_mismatches if sm['key'] == key), None)
                if size_mismatch_info:
                    actual_mb = size_mismatch_info['actual_size'] / (1024 * 1024)
                    if actual_mb >= 1024:
                        actual_str = f"{actual_mb/1024:.2f} GB"
                    else:
                        actual_str = f"{actual_mb:.2f} MB"
                    status = f" (size mismatch - re-downloading: current {actual_str} vs expected {size_str})"
                    logging.info(f"  - {key} {status}")
                else:
                    logging.info(f"  - {key} ({size_str})")
            logging.info("")  # Empty line for readability
        
        # Download files using thread pool
        if to_download:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._download_file, item) for item in to_download]
                completed = 0
                failed = 0
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        completed += 1
                    else:
                        failed += 1
                
            logging.info(f"Download summary:")
            logging.info(f"  - Successfully downloaded: {completed}")
            logging.info(f"  - Failed downloads: {failed}")
            if size_mismatches:
                logging.info(f"  - Size mismatches resolved: {len([sm for sm in size_mismatches if any(f.result() for f in futures)])}")
        
        # Handle cleanup if enabled
        if self.enable_cleanup:
            to_delete = [key for key in local_objects if key not in remote_objects]
            if to_delete:
                logging.info(f"Removing {len(to_delete)} local files...")
                for key in to_delete:
                    try:
                        os.remove(os.path.join(self.target_dir, key))
                        logging.info(f"Removed: {key}")
                    except OSError as e:
                        logging.error(f"Failed to remove {key}: {str(e)}")

def setup_logging(base_dir, debug=False):
    """Setup logging with timestamp-based log file"""
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'r2_sync_{timestamp}.log')
    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Disable noisy loggers
    for logger in ['boto3', 'botocore', 'urllib3']:
        logging.getLogger(logger).setLevel(logging.WARNING)
    
    return log_file

def parse_args():
    parser = argparse.ArgumentParser(description='Sync models from Cloudflare R2')
    parser.add_argument('--account-id',
                       help='Cloudflare R2 Account ID (env: R2_ACCOUNT_ID)',
                       default=os.getenv('R2_ACCOUNT_ID'))
    parser.add_argument('--access-key',
                       help='Cloudflare R2 Access Key ID (env: R2_ACCESS_KEY_ID)',
                       default=os.getenv('R2_ACCESS_KEY_ID'))
    parser.add_argument('--secret-key',
                       help='Cloudflare R2 Secret Access Key (env: R2_SECRET_ACCESS_KEY)',
                       default=os.getenv('R2_SECRET_ACCESS_KEY'))
    parser.add_argument('--bucket',
                       help='R2 Bucket name (env: R2_BUCKET_NAME)',
                       default=os.getenv('R2_BUCKET_NAME', 'static-libnnc'))
    parser.add_argument('--path', '-p', 
                       help='Target directory for downloads (default: ./models)',
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    parser.add_argument('--cleanup', '-c',
                       help='Remove local files that don\'t exist in R2',
                       action='store_true')
    parser.add_argument('--workers', '-w',
                       help='Number of concurrent downloads (default: 5)',
                       type=int, default=5)
    parser.add_argument('--retries', '-r',
                       help='Maximum number of retries for failed downloads (default: 3)',
                       type=int, default=3)
    parser.add_argument('--size-tolerance', '-t',
                       help='File size tolerance in MB for skipping downloads (default: 10)',
                       type=int, default=10)
    parser.add_argument('--exclude',
                       help='Skip files whose name ends with this suffix (default: i8x.ckpt)',
                       default='i8x.ckpt')
    parser.add_argument('--debug', '-d',
                       help='Enable debug logging',
                       action='store_true')
    parser.add_argument('--schedule', '-s',
                       help='Run as a scheduled task every 24 hours',
                       action='store_true')
    return parser.parse_args()
    
def main(args):
    # Verify required credentials are provided
    required_params = {
        'Account ID': args.account_id,
        'Access Key': args.access_key,
        'Secret Key': args.secret_key,
        'Bucket Name': args.bucket
    }
    
    missing_params = [param for param, value in required_params.items() if not value]
    if missing_params:
        print(f"Error: Missing required parameters: {', '.join(missing_params)}.")
        print("Provide them via command line arguments or environment variables.")
        print("\nEnvironment variables:")
        print("  R2_ACCOUNT_ID")
        print("  R2_ACCESS_KEY_ID") 
        print("  R2_SECRET_ACCESS_KEY")
        print("  R2_BUCKET_NAME (optional, defaults to 'static-libnnc')")
        sys.exit(1)
    
    os.makedirs(args.path, exist_ok=True)
    log_file = setup_logging(args.path, args.debug)
    
    logging.info(f"Starting sync to: {os.path.abspath(args.path)}")
    
    syncer = R2ModelSync(
        args.account_id,
        args.access_key,
        args.secret_key,
        args.bucket,
        args.path,
        enable_cleanup=args.cleanup,
        max_retries=args.retries,
        size_tolerance_mb=args.size_tolerance,
        exclude_suffix=args.exclude
    )
    
    syncer.sync(max_workers=args.workers)

def scheduled_task():
    args = parse_args() 
    """Wrapper function to run main() as a scheduled task"""
    os.makedirs(args.path, exist_ok=True)
    log_file = setup_logging(args.path, args.debug)

    try:
        logging.info(f"Starting scheduled sync at {datetime.now()}")
        main(args)
        logging.info(f"Completed scheduled sync at {datetime.now()}")
    except Exception as e:
        logging.error(f"Error in scheduled sync: {str(e)}")

def run_scheduler():
    schedule.every(36000).seconds.do(scheduled_task)
    
    while True:
        next_run = schedule.next_run()
        if next_run is not None:
            sleep_seconds = (next_run - datetime.now()).total_seconds()
            sleep_seconds = min(sleep_seconds, 36000)  # 10 hours
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        schedule.run_pending()

if __name__ == "__main__":
    args = parse_args()
    if args.schedule:
        print("====Starting scheduler mode====\n")
        # Run first sync immediately
        scheduled_task()
        # Then start the scheduler
        run_scheduler()
    else:
        # Normal single run
        print(f"====Starting normal sync ====\n")
        main(args)