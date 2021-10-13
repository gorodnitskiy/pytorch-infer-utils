import numpy as np
import onnx
import onnxruntime
import torch


def check_onnx(onnx_path: str) -> None:
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def check_onnx_complex(
    onnx_path: str,
    test_tensor: torch.Tensor,
    torch_outs: torch.Tensor,
    rtol: float = 1e-03,
    atol: float = 1e-05,
) -> None:
    # ordinary check
    check_onnx(onnx_path)

    # onnxruntime output check
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)

    if len(ort_outs) == 1:
        ort_outs = ort_outs[0]
    for ort_out, torch_out in zip(ort_outs, torch_outs):
        np.testing.assert_allclose(
            to_numpy(torch_out), ort_out, rtol=rtol, atol=atol
        )
