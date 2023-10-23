import pathlib
import subprocess
import types
from dataclasses import dataclass

from . import mod


@dataclass(kw_only=True)
class FileStatus:
    last_commit_hash: str | None
    modified_since_last_commit: bool | None
    filepath: str
    """relative to root path"""
    module: str
    root_module: str
    root_path: str


def _exec_git_cmd(start_path: str | pathlib.Path, cmd: list[str]):
    if isinstance(start_path, pathlib.Path):
        start_path = str(start_path)
    cmd = ["git", "-C", start_path, *cmd]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip(), None
    except subprocess.CalledProcessError as err:
        return None, err


def get_status_from_name(name: str):
    return get_status_from_module(mod.get_module_by_name(name))


def get_status_from_module(module: types.ModuleType):
    root = mod.get_root_module(module)
    root_path = mod.get_module_path(root)[0]
    root_path = pathlib.Path(root_path).parent

    filepath = module.__file__
    if filepath is None:
        raise NotImplementedError  # TODO:

    filepath = pathlib.Path(filepath)
    filepath = filepath.relative_to(root_path)

    hash, _ = _exec_git_cmd(
        root_path, ["log", "-1", "--format=%H", "--", str(filepath)]
    )
    if hash is None:
        modified = None
    else:
        m, _ = _exec_git_cmd(root_path, ["diff", "--name-only", str(filepath)])
        modified = None if m is None else "" != m

    st = FileStatus(
        last_commit_hash=hash,
        modified_since_last_commit=modified,
        filepath=str(filepath),
        module=module.__name__,
        root_module=root.__name__,
        root_path=str(root_path),
    )
    return st
