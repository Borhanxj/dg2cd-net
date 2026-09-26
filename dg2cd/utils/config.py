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


def apply_overrides(cfg: Config, overrides) -> Config:
    """Apply dotted KEY=VALUE overrides, e.g. "episodic.lr=0.01".

    Values are parsed as YAML, so 16 -> int, 0.01 -> float, true -> bool,
    [2,3] -> list. Unknown keys raise: a typo like "episodic.lrr=0.1" must
    fail loudly rather than silently leave the real setting unchanged.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        *parents, leaf = key.strip().split(".")
        node = cfg
        for part in parents:
            if not isinstance(getattr(node, part, None), Config):
                raise KeyError(f"unknown config section in {key!r}: {part!r}")
            node = getattr(node, part)
        if not hasattr(node, leaf):
            raise KeyError(f"unknown config key: {key!r}")
        setattr(node, leaf, Config._wrap(yaml.safe_load(raw)))
    return cfg