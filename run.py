#!/usr/bin/env python3
"""
Entry point for running the mnemo OCR pipeline.

Usage:
    python run.py
    python run.py --config configs/config.yaml
"""

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.pipeline import process_all_images_to_excel


def main():
    parser = argparse.ArgumentParser(description="Run OCR pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    process_all_images_to_excel(cfg)


if __name__ == "__main__":
    main()
