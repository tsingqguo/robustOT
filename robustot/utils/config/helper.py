import pathlib
from os import path
from typing import Callable, Iterable, Type, TypeVar

C = TypeVar("C")


def _get_sources_from_namespace(ns: dict) -> Iterable[str]:
    source: str | Iterable[str] | None = ns.get("source")
    if source is not None:
        if isinstance(source, str):
            source = [source]
        for s in source:
            yield s


def from_file(
    config_fp: str, config_cls: Type[C], config: C | None = None
) -> C:
    print("[DEBUG] loading config from file:", config_fp)
    if not path.exists(config_fp):
        raise FileNotFoundError(f"config file {config_fp} not found")
    if config is not None and not isinstance(config, config_cls):
        raise TypeError(
            f"input must be ModelConfig, got {config.__class__.__name__}"
        )
    ext = pathlib.Path(config_fp).suffix[1:]
    if ext == "py":
        ns = {}
        exec(open(config_fp).read(), {}, ns)

        sources = _get_sources_from_namespace(ns)
        [from_file(s, config_cls, config) for s in sources]

        configure: Callable[[C], None] | None = ns.get("configure")
        if configure is not None:
            if config is None:
                config = config_cls()
            configure(config)
    else:
        raise NotImplementedError(f"unsupported config file extension: {ext}")
    return config


def from_file_unsafe(config_fp: str):
    """
    load unknown config from file
    source is disabled
    """
    print("[DEBUG] loading config from file:", config_fp)
    if not path.exists(config_fp):
        raise FileNotFoundError(f"config file {config_fp} not found")
    assert pathlib.Path(config_fp).suffix[1:] == "py"

    ns = {}
    exec(open(config_fp).read(), {}, ns)
    configure: Callable[[], None] | None = ns.get("configure")
    if configure is not None:
        config = configure()
    else:
        raise ValueError("No configure function found in namespace")
    return config
