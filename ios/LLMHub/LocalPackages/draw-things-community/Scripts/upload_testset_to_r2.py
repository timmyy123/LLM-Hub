#!/usr/bin/env python3

from __future__ import annotations

import argparse
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a TestSet export directory to Cloudflare R2.")
    parser.add_argument("--source", type=Path, required=True, help="Export directory, for example ~/TestSetExport.")
    parser.add_argument("--bucket", required=True, help="R2 bucket name.")
    parser.add_argument(
        "--account-id",
        default=os.environ.get("CLOUDFLARE_ACCOUNT_ID", "cd96f610b0bb2657da157aca332052ec"),
        help="Cloudflare account id for the R2 endpoint.",
    )
    parser.add_argument(
        "--access-key-id",
        default=os.environ.get("R2_ACCESS_KEY_ID", ""),
        help="R2 access key id. Can also come from R2_ACCESS_KEY_ID.",
    )
    parser.add_argument(
        "--secret-access-key",
        default=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        help="R2 secret access key. Can also come from R2_SECRET_ACCESS_KEY.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent upload workers.")
    return parser.parse_args()


def iter_files(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.is_file())


def content_type_for(path: Path) -> str | None:
    content_type, _ = mimetypes.guess_type(str(path))
    return content_type


def build_client(account_id: str, access_key_id: str, secret_access_key: str):
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"}),
    )


def upload_file(client, bucket: str, source_root: Path, file_path: Path) -> str:
    key = file_path.relative_to(source_root).as_posix()
    extra_args = {}
    content_type = content_type_for(file_path)
    if content_type:
        extra_args["ContentType"] = content_type
    client.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
    return key


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")
    if not args.access_key_id or not args.secret_access_key:
        raise SystemExit("Missing R2 access credentials. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY or pass flags.")

    files = iter_files(source)
    print(f"Uploading {len(files)} files from {source} to bucket {args.bucket}...")
    client = build_client(args.account_id, args.access_key_id, args.secret_access_key)

    uploaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(upload_file, client, args.bucket, source, path): path
            for path in files
        }
        for future in as_completed(futures):
            key = future.result()
            uploaded += 1
            if uploaded == len(files) or uploaded % 250 == 0:
                print(f"Uploaded {uploaded}/{len(files)}: {key}")

    print("Upload complete.")


if __name__ == "__main__":
    main()
