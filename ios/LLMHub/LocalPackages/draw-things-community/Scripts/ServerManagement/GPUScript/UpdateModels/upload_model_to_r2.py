#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path


class UploadProgress:
    def __init__(self, total_bytes: int):
        self.total_bytes = total_bytes
        self.seen_bytes = 0
        self.last_percent = -1
        self.lock = threading.Lock()

    def __call__(self, bytes_amount: int) -> None:
        with self.lock:
            self.seen_bytes += bytes_amount
            percent = int(self.seen_bytes * 100 / self.total_bytes) if self.total_bytes else 100
            if percent != self.last_percent:
                self.last_percent = percent
                uploaded = format_bytes(self.seen_bytes)
                total = format_bytes(self.total_bytes)
                print(f"\rUploading: {percent:3d}% ({uploaded} / {total})", end="", flush=True)


def format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{byte_count} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload one model file to Cloudflare R2 using multipart upload.")
    parser.add_argument("--file", type=Path, required=True, help="Local model file to upload.")
    parser.add_argument(
        "--key",
        help="R2 object key. Defaults to the local file name, optionally under --prefix.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional R2 key prefix when --key is not provided, for example models/.",
    )
    parser.add_argument(
        "--account-id",
        default=os.getenv("R2_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        help="Cloudflare R2 account ID. Can also come from R2_ACCOUNT_ID or CLOUDFLARE_ACCOUNT_ID.",
    )
    parser.add_argument(
        "--access-key",
        default=os.getenv("R2_ACCESS_KEY_ID"),
        help="R2 access key ID. Can also come from R2_ACCESS_KEY_ID.",
    )
    parser.add_argument(
        "--secret-key",
        default=os.getenv("R2_SECRET_ACCESS_KEY"),
        help="R2 secret access key. Can also come from R2_SECRET_ACCESS_KEY.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("R2_BUCKET_NAME", "static-libnnc"),
        help="R2 bucket name. Can also come from R2_BUCKET_NAME. Default: static-libnnc.",
    )
    parser.add_argument(
        "--content-type",
        default="application/octet-stream",
        help="Content-Type metadata for the object. Default: application/octet-stream.",
    )
    parser.add_argument(
        "--multipart-threshold-mb",
        type=int,
        default=64,
        help="Use multipart upload for files at least this large. Default: 64.",
    )
    parser.add_argument(
        "--multipart-chunk-mb",
        type=int,
        default=64,
        help="Multipart chunk size in MiB. Default: 64.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum concurrent multipart upload workers. Default: 8.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing R2 object when its size differs from the local file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned upload without writing to R2.")
    return parser.parse_args()


def normalize_key(file_path: Path, key: str | None, prefix: str) -> str:
    if key:
        return key.strip("/")
    prefix = prefix.strip("/")
    if prefix:
        return f"{prefix}/{file_path.name}"
    return file_path.name


def load_boto3():
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
        from botocore.exceptions import ClientError
    except ModuleNotFoundError as error:
        missing = error.name or "boto3"
        print(
            f"Missing Python package '{missing}'. Install the R2 upload dependencies with:\n"
            "  python3 -m pip install boto3",
            file=sys.stderr,
        )
        sys.exit(2)
    return boto3, TransferConfig, Config, ClientError


def build_client(account_id: str, access_key: str, secret_key: str):
    boto3, _, Config, _ = load_boto3()
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"}),
    )


def head_object(client, bucket: str, key: str, client_error_type) -> dict | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except client_error_type as error:
        status_code = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        error_code = error.response.get("Error", {}).get("Code")
        if status_code == 404 or error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name, value in (
            ("--account-id / R2_ACCOUNT_ID", args.account_id),
            ("--access-key / R2_ACCESS_KEY_ID", args.access_key),
            ("--secret-key / R2_SECRET_ACCESS_KEY", args.secret_key),
            ("--bucket / R2_BUCKET_NAME", args.bucket),
        )
        if not value
    ]
    if missing:
        raise SystemExit("Missing required R2 settings:\n  " + "\n  ".join(missing))

    file_path = args.file.expanduser().resolve()
    if not file_path.is_file():
        raise SystemExit(f"Local file does not exist: {file_path}")

    file_size = file_path.stat().st_size
    key = normalize_key(file_path, args.key, args.prefix)
    endpoint = f"https://{args.account_id}.r2.cloudflarestorage.com"

    print(f"Source:   {file_path}")
    print(f"Size:     {format_bytes(file_size)} ({file_size} bytes)")
    print(f"Endpoint: {endpoint}")
    print(f"Bucket:   {args.bucket}")
    print(f"Key:      {key}")

    if args.dry_run:
        print("Dry run only; no upload performed.")
        return

    _, TransferConfig, _, ClientError = load_boto3()
    client = build_client(args.account_id, args.access_key, args.secret_key)
    remote = head_object(client, args.bucket, key, ClientError)
    if remote is not None:
        remote_size = int(remote.get("ContentLength", -1))
        if remote_size == file_size:
            print(f"Remote object already exists with matching size: {format_bytes(remote_size)}")
            print("Skipping upload.")
            return
        if not args.overwrite:
            raise SystemExit(
                "Remote object already exists with a different size:\n"
                f"  remote: {format_bytes(remote_size)} ({remote_size} bytes)\n"
                f"  local:  {format_bytes(file_size)} ({file_size} bytes)\n"
                "Pass --overwrite to replace it."
            )
        print(f"Remote object exists with different size ({format_bytes(remote_size)}); overwriting.")

    chunk_size = args.multipart_chunk_mb * 1024 * 1024
    threshold = args.multipart_threshold_mb * 1024 * 1024
    transfer_config = TransferConfig(
        multipart_threshold=threshold,
        multipart_chunksize=chunk_size,
        max_concurrency=args.workers,
        use_threads=True,
    )

    progress = UploadProgress(file_size)
    client.upload_file(
        str(file_path),
        args.bucket,
        key,
        ExtraArgs={"ContentType": args.content_type},
        Config=transfer_config,
        Callback=progress,
    )
    print()

    uploaded = head_object(client, args.bucket, key, ClientError)
    if uploaded is None:
        raise SystemExit("Upload finished, but head-object could not find the uploaded object.")
    uploaded_size = int(uploaded.get("ContentLength", -1))
    if uploaded_size != file_size:
        raise SystemExit(
            "Upload finished, but remote size does not match local file:\n"
            f"  remote: {format_bytes(uploaded_size)} ({uploaded_size} bytes)\n"
            f"  local:  {format_bytes(file_size)} ({file_size} bytes)"
        )

    print("Upload complete and size verified.")
    etag = uploaded.get("ETag", "").strip('"')
    print(f"Remote ETag: {etag}")


if __name__ == "__main__":
    main()
