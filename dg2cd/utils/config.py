"""YAML config loading with attribute access."""
from pathlib import Path
from typing import Any
import yaml


class Config:
    """Recursive attribute-access wrapper around a nested dict."""

    def __init__(self, data: dict[str, Any]):
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        """Recover a plain dict -- for saving next to checkpoints."""
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: str | Path) -> Config:
    """Load a YAML config file.

    Args:
        path: path to the YAML file, e.g. "configs/pacs.yaml".

    Returns:
        Config with nested attribute access.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")

    return Config(data)