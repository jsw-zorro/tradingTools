"""Strategy plugins - auto-discovers and imports all submodules."""

import importlib
import pkgutil


def _auto_discover():
    """Import all subpackages so @register decorators fire."""
    package_path = __path__
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package_path, prefix=__name__ + "."
    ):
        try:
            importlib.import_module(modname)
        except Exception:
            pass


_auto_discover()
