"""Record inputs and outputs of named submodules with forward hooks, and replay them.

A recording is a flat dict of tensors plus a JSON-serializable list of calls. Each call stores the structure of the
positional arguments, keyword arguments and output, so a recorded module can be called again on its recorded inputs.
Tensors keep the dtype the module computed them in.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Self

import torch
from torch import nn


def flatten(obj: Any, key: str, tensors: dict[str, torch.Tensor]) -> Any:
    """Store the tensors of a nested structure in `tensors` and return a JSON spec of the structure."""
    if isinstance(obj, torch.Tensor):
        tensors[key] = obj.detach().to("cpu").clone().contiguous()
        return {"tensor": key}
    if obj is None:
        return {"none": True}
    if isinstance(obj, (bool, int, float, str)):
        return {"value": obj}
    if isinstance(obj, Mapping):
        return {"dict": {str(k): flatten(v, f"{key}.{k}", tensors) for k, v in obj.items()}}
    if isinstance(obj, (tuple, list)):
        kind = "tuple" if isinstance(obj, tuple) else "list"
        return {kind: [flatten(v, f"{key}.{i}", tensors) for i, v in enumerate(obj)]}
    raise TypeError(f"cannot record object of type {type(obj).__name__} at {key}")


def unflatten(spec: Any, tensors: Mapping[str, Any]) -> Any:
    """Rebuild a structure recorded by `flatten`. Tensors may be torch tensors or NumPy arrays."""
    if "tensor" in spec:
        value = tensors[spec["tensor"]]
        return value if isinstance(value, torch.Tensor) else torch.from_numpy(value.copy())
    if "none" in spec:
        return None
    if "value" in spec:
        return spec["value"]
    if "dict" in spec:
        return {k: unflatten(v, tensors) for k, v in spec["dict"].items()}
    if "tuple" in spec:
        return tuple(unflatten(v, tensors) for v in spec["tuple"])
    if "list" in spec:
        return [unflatten(v, tensors) for v in spec["list"]]
    raise ValueError(f"bad spec {spec}")


class ActivationRecorder:
    """Context manager recording the calls of `module_names` in `model`.

    Keys are `inputs.<tag>.<module>@<call>.args...`, `inputs.<tag>.<module>@<call>.kwargs...` and
    `outputs.<tag>.<module>@<call>.out...`, where `<call>` counts calls of the same module.
    """

    def __init__(self, model: nn.Module, module_names: Iterable[str], tag: str = "modules") -> None:
        modules = dict(model.named_modules())
        missing = [name for name in module_names if name not in modules]
        if missing:
            raise KeyError(f"modules not found: {missing}")
        self.model = model
        self.module_names = list(module_names)
        self.modules = modules
        self.tag = tag
        self.tensors: dict[str, torch.Tensor] = {}
        self.calls: list[dict] = []
        self._counts: dict[str, int] = defaultdict(int)
        self._handles: list = []

    def _hook(self, name: str):
        def hook(module, args, kwargs, output):
            call = self._counts[name]
            self._counts[name] += 1
            base = f"{self.tag}.{name}@{call}"
            self.calls.append(
                {
                    "module": name,
                    "call": call,
                    "class": type(module).__name__,
                    "args": flatten(tuple(args), f"inputs.{base}.args", self.tensors),
                    "kwargs": flatten(dict(kwargs), f"inputs.{base}.kwargs", self.tensors),
                    "output": flatten(output, f"outputs.{base}.out", self.tensors),
                }
            )

        return hook

    def __enter__(self) -> Self:
        for name in self.module_names:
            self._handles.append(self.modules[name].register_forward_hook(self._hook(name), with_kwargs=True))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def replay(model: nn.Module, call: dict, tensors: Mapping[str, Any]) -> Any:
    """Call the recorded module of `call` on its recorded inputs and return the new output."""
    module = dict(model.named_modules())[call["module"]]
    args = unflatten(call["args"], tensors)
    kwargs = unflatten(call["kwargs"], tensors)
    with torch.no_grad():
        return module(*args, **kwargs)


def recorded_output(call: dict, tensors: Mapping[str, Any]) -> Any:
    return unflatten(call["output"], tensors)


def iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
    """Yield the tensors of a nested structure in a stable order."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            yield from iter_tensors(value)


def modules_of_classes(model: nn.Module, class_names: Iterable[str]) -> list[str]:
    """Names of submodules whose class name is in `class_names`, in registration order."""
    wanted = set(class_names)
    return [name for name, module in model.named_modules() if name and type(module).__name__ in wanted]


def library_versions() -> dict:
    import diffusers
    import numpy
    import transformers

    return {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
    }
