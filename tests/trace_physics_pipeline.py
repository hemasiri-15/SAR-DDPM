import inspect
import torch
import sys
import torch.nn as nn

executed = []

def hook(name):
    def _hook(module, inp, out):
        executed.append(name)
    return _hook


for mod in list(nn.Module.__subclasses__()):
    pass


import structdiff

registered = []

for module in list(sys.modules.values()):
    if module is None:
        continue

    for name in dir(module):
        try:
            obj = getattr(module, name)
        except Exception:
            continue

        if inspect.isclass(obj):
            try:
                if issubclass(obj, nn.Module):
                    registered.append(obj)
            except Exception:
                pass

keywords = [
    "Physics",
    "Orientation",
    "Statistic",
    "Relation",
    "Bias",
    "Attention",
]

hooks = []

for cls in registered:
    if any(k.lower() in cls.__name__.lower() for k in keywords):
        try:
            old_init = cls.__init__

            def make_new_init(old_init, cls):
                def new_init(self, *a, **kw):
                    old_init(self, *a, **kw)
                    self.register_forward_hook(hook(cls.__name__))
                return new_init

            cls.__init__ = make_new_init(old_init, cls)

        except Exception:
            pass

print("Hooks installed.")
