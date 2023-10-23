import importlib
import sys
import types

from .log import get_logger

log = get_logger(__name__)


def get_module_by_name(
    name: str, else_import: bool = False
) -> types.ModuleType:
    if name not in sys.modules:
        if else_import:
            importlib.import_module(name)
        else:
            raise KeyError(f"module {name} not found")
    return sys.modules[name]


def get_module_path(module: types.ModuleType):
    return module.__path__


def get_root_module(module: types.ModuleType) -> types.ModuleType:
    root_name = module.__name__.split(".")[0]
    return get_module_by_name(root_name, else_import=True)


def safer_sys_path_append(module: types.ModuleType) -> str:
    try:
        mod_path = get_module_path(module)[0]
    except AttributeError:
        log.critical("failed to resolve path of module {}".format(module))
        exit(1)
    if mod_path not in sys.path:
        sys.path.append(mod_path)
        log.debug('appended "{}" to sys.path'.format(mod_path))
    return mod_path
