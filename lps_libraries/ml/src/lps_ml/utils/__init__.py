"""Basic __init__.py
Allows to import all the content defined in __all__ in the files inside the folder by folder name.
"""
import importlib
import pkgutil

__all__ = []

for module_info in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_info.name}")
    for name in getattr(module, "__all__", []):
        globals()[name] = getattr(module, name)
        __all__.append(name)
