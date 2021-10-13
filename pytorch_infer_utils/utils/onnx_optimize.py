from typing import List, Optional

import onnx
from onnxoptimizer import get_available_passes, optimize
from onnxsim import simplify


def optimize_onnx(
    path_in: str,
    path_out: Optional[str] = None,
    passes: Optional[List[str]] = None,
    fixed_point: bool = False,
) -> str:
    if path_out is None:
        path_out = path_in.replace(".onnx", "_opt.onnx")

    model = onnx.load(path_in)
    available_passes = get_available_passes()
    passes = [] if passes is None else passes
    passes = [p for p in passes if p in available_passes]
    model = optimize(model, passes, fixed_point=fixed_point)

    onnx.save(model, path_out)
    return path_out


def optimize_onnxsim(
    path_in: str, path_out: Optional[str] = None, **kwargs
) -> str:
    if path_out is None:
        path_out = path_in.replace(".onnx", "_sim.onnx")

    model = onnx.load(path_in)
    model_simple, check = simplify(model, **kwargs)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model, path_out)
    return path_out
