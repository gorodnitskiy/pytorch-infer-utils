from typing import Any, Dict, List, Optional

import pkg_resources
import torch

from .utils import (
    check_onnx,
    check_onnx_complex,
    optimize_onnx,
    optimize_onnxsim,
    yaml_parser,
)

_ONNX_BASE_CFG_PATH = "config/onnx_cfg.yaml"


def load_weights_rounded_model(
    model: torch.nn.Module,
    state_dict_path: str,
    n_digits: int = 16,
    save_dict_path: Optional[str] = None,
    map_location: Optional[str] = None,
) -> None:
    """
    Sometimes PyTorch model has denormal weights and
    the onnx-model has a mush slower execution speed
    (via onnxruntime) than expected. Manually weights
    rounding is intensively increasing speed: in my case
    there is 225 -> 75 ms per image via onnxruntime. And
    it didn't affect to the output probabilities.
    To solve the problem: manually rounds weights (n_digits = 16).
    """
    state_dict = torch.load(state_dict_path, map_location=map_location)
    if n_digits:
        for key, value in state_dict.items():
            upd_value = torch.tensor(value.numpy(), dtype=torch.float32)
            upd_value = upd_value * 10 ** n_digits
            state_dict[key] = upd_value.round() / (10 ** n_digits)
        if save_dict_path:
            torch.save(state_dict, save_dict_path)

    model.load_state_dict(state_dict)
    model.eval()


class ONNXExporter:
    input_axis = ("B", "C", "H", "W")

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        :param cfg: config with params:
            - default_shapes: test tensor default shape for exporting,
                default = [1, 3, 224, 224]
            - onnx_export_params: onnx exporting params:
                - export_params: store the trained parameter weights
                    inside the model file, default = True
                - do_constant_folding: whether to execute constant folding
                    for optimization, default = True
                - input_names: the model's input names, default = ["input"]
                - output_names: the model's output names, default = ["output"]
                - opset_version: the ONNX version to export the model to,
                    default = 11
            - onnx_optimize_params: onnx optimizing params:
                - fixed_point: use fixed_point, default = False
                - passes: passes list to use, default = [...]
        :param kwargs: additional params for replace default values in cfg
        """
        if cfg is None:
            cfg = yaml_parser(
                pkg_resources.resource_filename(__name__, _ONNX_BASE_CFG_PATH)
            )
            for key, value in kwargs.items():
                if key in cfg:
                    cfg[key] = value
                elif key in cfg["onnx_export_params"]:
                    cfg["onnx_export_params"][key] = value
                elif key in cfg["onnx_optimize_params"]:
                    cfg["onnx_optimize_params"][key] = value

        self.default_shapes = cfg["default_shapes"]
        self.export_params = cfg["onnx_export_params"]
        self.optimize_params = cfg["onnx_optimize_params"]
        self._dynamic_axes, self._torch_outs = None, None
        self.test_tensor = None

    def set_torch_outs(self, torch_outs: torch.Tensor) -> None:
        self._torch_outs = torch_outs

    def set_dynamic_axes(self, input_shapes: List[int]) -> None:
        size_error_msg = f"support only {self.input_axis} configuration"
        assert len(self.input_axis) == len(input_shapes), size_error_msg

        input_shapes_reduced = input_shapes
        input_name = self.export_params["input_names"][0]
        output_name = self.export_params["output_names"][0]

        for i, dim_name in enumerate(self.input_axis):
            if input_shapes_reduced[i] == -1:
                if not isinstance(self._dynamic_axes, dict):
                    self._dynamic_axes = dict()
                    self._dynamic_axes[input_name] = dict()
                    self._dynamic_axes[output_name] = dict()

                input_shapes_reduced[i] = self.default_shapes[i]
                self._dynamic_axes[input_name][i] = dim_name
                self._dynamic_axes[output_name][i] = dim_name

        self.test_tensor = torch.randn(
            *input_shapes_reduced, requires_grad=True
        )

    def torch2onnx(
        self,
        torch_model: torch.nn.Module,
        onnx_path_out: str,
        input_shapes: List[int],
    ) -> str:
        if ".onnx" in onnx_path_out:
            onnx_path_out = onnx_path_out.replace(".onnx", "")
        opset_version = self.export_params["opset_version"]
        onnx_path_out = f"{onnx_path_out}_op{opset_version}.onnx"

        self.set_dynamic_axes(input_shapes)
        self.set_torch_outs(torch_model(self.test_tensor))
        torch.onnx.export(
            torch_model,
            self.test_tensor,
            onnx_path_out,
            dynamic_axes=self._dynamic_axes,
            **self.export_params,
        )

        check_onnx_complex(onnx_path_out, self.test_tensor, self._torch_outs)
        return onnx_path_out

    def optimize_onnx(
        self, onnx_path_in: str, onnx_path_out: Optional[str] = None
    ) -> str:
        opt_onnx_path = optimize_onnx(
            onnx_path_in, onnx_path_out, **self.optimize_params
        )
        if self._torch_outs is not None:
            check_onnx_complex(
                opt_onnx_path, self.test_tensor, self._torch_outs
            )
        else:
            check_onnx(opt_onnx_path)

        return opt_onnx_path

    def optimize_onnx_sim(
        self, onnx_path_in: str, onnx_path_out: Optional[str] = None, **kwargs
    ) -> str:
        opt_onnx_path = optimize_onnxsim(onnx_path_in, onnx_path_out, **kwargs)
        if self._torch_outs is not None:
            check_onnx_complex(
                opt_onnx_path, self.test_tensor, self._torch_outs
            )
        else:
            check_onnx(opt_onnx_path)

        return opt_onnx_path

    def torch2optimized_onnx(
        self,
        torch_model: torch.nn.Module,
        onnx_path_out: str,
        input_shapes: List[int],
        **kwargs,
    ) -> List[str]:
        onnx_path_out = self.torch2onnx(
            torch_model, onnx_path_out, input_shapes
        )
        opt_onnx_path = self.optimize_onnx(onnx_path_out)
        sim_onnx_path = self.optimize_onnx_sim(opt_onnx_path, **kwargs)
        return [onnx_path_out, opt_onnx_path, sim_onnx_path]
