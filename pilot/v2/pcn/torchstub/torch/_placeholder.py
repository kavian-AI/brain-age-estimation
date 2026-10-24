"""Placeholder for torch, used ONLY so pcntoolkit 0.35 can import its neural-process / RFA
modules, which the BLR braincharts never use. Every name is a class that can be subclassed,
but instantiating or calling anything raises TorchPlaceholderUsed. So a run that completes
is proof that torch was never actually needed."""
class TorchPlaceholderUsed(RuntimeError):
    pass

class _Meta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return make(name)

class _Base(metaclass=_Meta):
    def __init__(self, *a, **k):
        raise TorchPlaceholderUsed(f"torch placeholder '{type(self).__name__}' was used: torch is really needed")

_cache = {}
def make(name):
    if name not in _cache:
        _cache[name] = _Meta(name, (_Base,), {})
    return _cache[name]
