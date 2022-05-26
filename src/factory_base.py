from attrdict import AttrDict
import importlib
import pkgutil


class FactoryBase:
    def __init__(self, package: str, conf: AttrDict):
        # scope to relevant section in config
        self._conf = conf
        for attr in package.split('.'):
            if attr != 'src':
                self._conf = self._conf(attr)

        # get package from its name
        package = importlib.import_module(package)

        # fill library of implementations
        self._implementations = {}
        for loader, name, _ in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + '.' + name
            self._implementations[name] = importlib.import_module(full_name)

    def list_implementations(self):
        """List all available implementations for this factory.

        :return: List of all available implementations
        """
        return [k for k, v in self._implementations.items() if hasattr(v, k)]

    def get_implementation(self, key, conf_profile=0):
        """Gets an implementation by its name. Throws RuntimeError if implementation is not available.

        :param conf_profile: The configuration profile (index) to use for this implementation
        :param key: The name of the desired implementation
        :return: The desired implementation
        """
        if key not in self._implementations:
            raise RuntimeError(f'Could not find a module named {key}')
        mod = self._implementations[key]
        impl_cls = getattr(mod, key, None)
        if not impl_cls:
            raise RuntimeError(f'Could not find class {key} in module {mod}')
        return impl_cls(self._conf(key)[conf_profile] if key in self._conf else AttrDict({}))
