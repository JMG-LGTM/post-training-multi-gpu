#!/usr/bin/env python3
"""Package explicitly same-state preference pairs; never infer labels or alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Input same_state_action_preferences_v1 JSON; relative NPZ paths are resolved beside this file")
    parser.add_argument("--output", type=Path, required=True, help="New portable package directory; existing paths are refused")
    args = parser.parse_args()
    from post_training.preference_data import package_preferences
    print(json.dumps(package_preferences(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
