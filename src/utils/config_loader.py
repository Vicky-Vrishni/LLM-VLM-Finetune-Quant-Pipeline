"""
Config Loader Utility
----------------------
Ye module YAML config files ko Python dictionary me load karta hai.
Saare training/quantization scripts isi function ko use karenge,
taaki config reading ka logic ek hi jagah rahe (DRY principle).
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML config file ko padh kar Python dictionary return karta hai.

    Args:
        config_path: YAML file ka path (e.g. "configs/llm_finetune.yaml")

    Returns:
        Dictionary jisme config ki saari values hongi.

    Raises:
        FileNotFoundError: agar config file exist nahi karti.
        yaml.YAMLError: agar YAML file me syntax error hai.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file nahi mili: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parsing error in {config_path}: {e}")

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Dictionary ko YAML file me save karta hai.
    Useful hai jab hum modified config (e.g. quantization report ke saath)
    wapas disk pe likhna chahein.

    Args:
        config: Dictionary jo save karni hai.
        output_path: Output YAML file ka path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Quick test - jab ye file directly run karoge to ye check karega
    # ki config sahi se load ho rahi hai ya nahi.
    test_config = load_config("configs/llm_finetune.yaml")
    print("Config loaded successfully:")
    print(test_config)