#!/usr/bin/env python3
"""Train the custom PixelIQL preference scorer, not author-official IDQL.

This legacy experiment has no behavior actor optimizer and does not use the
separate official JAX core in train_idql_official.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Verified native transition package")
    parser.add_argument("--preferences", type=Path, help="Override the explicit same-state pair package")
    parser.add_argument("--frozen-members", type=Path, help="Optional original full native split roster")
    parser.add_argument("--dry-run", action="store_true", help="Validate identities and counts without CUDA or model loading")
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint from the same output directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    from post_training.preference_config import load_config, validate_preference_config
    from post_training.preference_training import dry_run, train
    config = load_config(args.config, preferences=args.preferences, frozen_members=args.frozen_members)
    for field in ("output", "cache"):
        if getattr(args, field):
            config[field + "_dir"] = str(getattr(args, field))
    validate_preference_config(config)
    if args.dry_run:
        if int(os.environ.get("RANK", "0")) == 0:
            result = dry_run(config, args.data, int(os.environ.get("WORLD_SIZE", "1")))
            print(json.dumps(result, indent=2, allow_nan=False))
            if not result["environment"]["ready"]:
                raise SystemExit(2)
        return
    train(config, args.data, resume=args.resume)


if __name__ == "__main__":
    main()
