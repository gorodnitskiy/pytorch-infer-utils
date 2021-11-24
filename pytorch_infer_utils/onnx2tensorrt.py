from typing import Any, Callable, Dict, List, Optional

import onnx
import pkg_resources

try:
    import tensorrt as trt
except Exception as exception:
    print(exception)

import torch
from advanced_argparse import PrettySafeLoader, yaml_parser

from .utils import BatchStream, EntropyCalibrator, check_tensorrt_health

trtLogLevel = trt.Logger.Severity

_TENSORRT_CFG_PATH = "config/tensorrt_cfg.yaml"


def save_engine(engine: trt.ICudaEngine, path: str) -> None:
    serialized = bytearray(engine.serialize())
    with open(path, "wb") as engine_file:
        engine_file.write(serialized)


def load_engine(
    path: str, log_level: trtLogLevel = trt.Logger.INFO
) -> trt.ICudaEngine:
    with trt.Logger(log_level) as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as engine_file:
            engine_data = engine_file.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine


class TRTEngineBuilder:
    def __init__(
        self,
        use_opt_shapes: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
        log_level: trtLogLevel = trt.Logger.INFO,
        **kwargs,
    ) -> None:
        """
        :param use_opt_shapes: use optimal shapes config option, default = False
        :param cfg: config with params:
            - opt_shape_dict: optimal shapes dict (minimal, average, maximal),
                default = {'input': [[1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224]]}
            - max_workspace_size: default = [1, 30] or 1 << 30
            - max_batch_size: default = 1
            - cache_file: cache filename for int8_mode, default = "model.trt.int8calibration"
        :param log_level: tensorrt logging level, default = trt.Logger.INFO
        :param kwargs: additional params for replace default values in cfg
        """
        check_tensorrt_health()
        self._use_opt_shapes = use_opt_shapes
        if cfg is None:
            cfg = yaml_parser(
                pkg_resources.resource_filename(__name__, _TENSORRT_CFG_PATH),
                loader=PrettySafeLoader,
            )
            for key, value in kwargs.items():
                if key in cfg:
                    cfg[key] = value
                elif key in cfg["opt_shape_dict"]:
                    self._use_opt_shapes = True
                    cfg["opt_shape_dict"][key] = value

        self._cfg = cfg
        self.logger = trt.Logger(log_level)

        # create builder and network
        self.builder = trt.Builder(self.logger)
        explicit_batch = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
        self.network = self.builder.create_network(explicit_batch)
        self.config = None

    def set_builder_config(self, **kwargs) -> None:
        self.config = self.builder.create_builder_config(**kwargs)

    def build_engine(
        self,
        onnx_path: str,
        fp16_mode: bool = False,
        int8_mode: bool = False,
        gpu_device_id: int = 0,
        max_batch_size: int = 1,
        calibration_set: Optional[List[Any]] = None,
        max_image_shape: Optional[List[int]] = None,
        load_image_func: Optional[Callable] = None,
        engine_path: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> trt.ICudaEngine:
        # parse onnx
        parser = trt.OnnxParser(self.network, self.logger)
        onnx_model = onnx.load(onnx_path)

        if not parser.parse(onnx_model.SerializeToString()):
            error_msgs = ""
            for error in range(parser.num_errors):
                error_msgs += f"{parser.get_error(error)}\n"
            raise RuntimeError(f"parse onnx failed:\n{error_msgs}")

        # config builder
        max_workspace_size = self._cfg["max_workspace_size"]
        max_workspace_size = max_workspace_size[0] << max_workspace_size[1]
        self.builder.max_batch_size = max_batch_size
        self.builder.max_workspace_size = max_workspace_size

        self.set_builder_config(**kwargs)
        self.config.max_workspace_size = max_workspace_size
        profile = self.builder.create_optimization_profile()

        if self._use_opt_shapes:
            for input_name, param in self._cfg["opt_shape_dict"].items():
                min_shape = tuple(param[0][:])
                opt_shape = tuple(param[1][:])
                max_shape = tuple(param[2][:])
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            self.config.add_optimization_profile(profile)

        if fp16_mode:
            self.builder.fp16_mode = fp16_mode
            self.config.set_flag(trt.BuilderFlag.FP16)

        if int8_mode:
            msg = "INT8 mode requires calibration_set"
            assert calibration_set is not None, msg

            self.builder.int8_mode = int8_mode
            stream = BatchStream(
                images=calibration_set,
                batch_size=self._cfg["stream_batch_size"],
                max_image_shape=max_image_shape,
                load_image_func=load_image_func,
                verbose=verbose,
            )

            calibrator = EntropyCalibrator(
                stream=stream,
                cache_file=self._cfg["cache_file"],
            )
            self.builder.int8_calibrator = calibrator
            self.config.set_flag(trt.BuilderFlag.INT8)

        # create engine
        device = torch.device(f"cuda:{gpu_device_id}")
        with torch.cuda.device(device):
            if int8_mode:
                engine = self.builder.build_cuda_engine(self.network)
            else:
                engine = self.builder.build_engine(self.network, self.config)

        if engine_path:
            save_engine(engine, engine_path)

        return engine
