#!/usr/bin/env python3
"""Download models from ModelScope.cn."""

import argparse
import os
import sys

from modelscope import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--out", default="./models")
    p.add_argument("--pat", default=None)
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()

    if args.dry:
        print(f"Repo: {args.repo}")
        print("Use --pat to filter. Examples:")
        print("  --pat 'config.json'         # config only")
        print("  --pat '*.safetensors'       # weights")
        print("  --pat 'model-00001*'        # first shard")
        return

    print(f"Downloading {args.repo} -> {args.out}")
    try:
        path = snapshot_download(
            args.repo,
            cache_dir=args.out,
            allow_file_pattern=args.pat,
        )
        print(f"Done: {path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
