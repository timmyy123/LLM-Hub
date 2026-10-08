#!/usr/bin/env python3
"""
Compute SHA256 hash of first 4KB + last 4KB of a file.

Usage: python3 8k_hash.py <file_path>

For files <= 8KB, hashes the entire file.
Output: Just the hex digest (compatible with sha256sum output format).
"""

import hashlib
import os
import sys


def compute_8k_hash(file_path):
    """Compute SHA256 of first 4KB + last 4KB of a file."""
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        if file_size <= 8192:
            data = f.read()
        else:
            first_4k = f.read(4096)
            f.seek(-4096, 2)
            last_4k = f.read(4096)
            data = first_4k + last_4k

    return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file_path>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file", file=sys.stderr)
        sys.exit(1)

    print(compute_8k_hash(file_path))
