#!/usr/bin/env python3
"""CLI script to trigger model training via the REST API."""

from __future__ import annotations

import argparse
import sys

import requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigger hybrid recommender training."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the raw interaction CSV on the server.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the recommender API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if models already exist.",
    )
    args = parser.parse_args()

    url = f"{args.api_url}/train"
    payload = {"data_path": args.data_path, "force_retrain": args.force}

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        print(f"✓ {resp.json()}")
    except requests.RequestException as exc:
        print(f"✗ Training request failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
