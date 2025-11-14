"""
Загрузка конфигурационного файла config.yaml
"""

import yaml
from pathlib import Path


def load_config(path: str | Path = "config.yaml") -> dict:
    cfg_path = Path(path).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)