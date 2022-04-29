from attrdict import AttrDict
import importlib
import pkgutil


class ImplementationFinder:
    def __init__(self, package: str, conf: AttrDict):
        package = importlib.import_module(package)
        self._implementations = {}
        self._conf = conf
        for attr in package.split('.'):
            self._conf = self._conf(attr)
        for loader, name, _ in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + '.' + name
            self._implementations[name] = importlib.import_module(full_name)

    def list_implementations(self):
        return list(self._implementations.keys())

    def get_implementation(self, key):
        if key not in self._implementations:
            raise RuntimeError(f'Could not find a module named {key}')
        mod = self._implementations[key]
        impl_cls = getattr(mod, key, None)
        if not impl_cls:
            raise RuntimeError(f'Could not find class {key} in module {mod}')
        return impl_cls(self._conf(key) if key in self._conf else AttrDict({}))
