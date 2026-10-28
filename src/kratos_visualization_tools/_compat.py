"""Live module facades for the historical standalone import APIs."""

from types import ModuleType


class _CompatibilityModule(ModuleType):
    def __getattribute__(self, name):
        namespace = ModuleType.__getattribute__(self, "__dict__")
        owners = namespace.get("_compat_owners", {})
        if name in owners:
            return getattr(owners[name], name)
        return ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        namespace = ModuleType.__getattribute__(self, "__dict__")
        owners = namespace.get("_compat_owners", {})
        if name in owners:
            # Update both the definition and every module that imported it.
            # This also makes monkeypatch restoration propagate symmetrically.
            for module in namespace["_compat_modules"]:
                if name in vars(module):
                    setattr(module, name, value)
        ModuleType.__setattr__(self, name, value)


def install_facade(module, owners, modules):
    """Expose legacy names while keeping calls and mutable state live."""
    module._compat_owners = owners
    module._compat_modules = modules
    for name, owner in owners.items():
        setattr(module, name, getattr(owner, name))
    module.__all__ = [name for name in owners if not name.startswith("_")]
    module.__class__ = _CompatibilityModule
