from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from .onnx2tensorrt import TRTEngineBuilder, load_engine


class HostDeviceObj:
    def __init__(self, host: Any, device: Any, size: Any) -> None:
        self.host = host
        self.device = device
        self.size = size

    def __str__(self) -> str:
        return (
            f"Host: {str(self.host)}\n"
            + f"Device: {str(self.device)}\n"
            + f"Size: {str(self.size)}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TRTRunWrapper:
    # noinspection PyArgumentList
    def __init__(self, trt_path: Union[str, trt.ICudaEngine]) -> None:
        if isinstance(trt_path, str):
            self.engine = load_engine(trt_path)
        else:
            self.engine = trt_path
        self.stream = cuda.Stream()
        self.inputs, self.outputs, self.bindings = self._alloc_buffers_(
            self.engine
        )

    @staticmethod
    def _alloc_buffers_(engine: trt.ICudaEngine) -> Tuple[List[Any], ...]:
        inputs = list()
        outputs = list()
        bindings = list()
        for binding in engine:
            b_size = engine.get_binding_shape(binding)
            b_type = engine.get_binding_dtype(binding)
            host_mem = cuda.pagelocked_empty(
                trt.volume(b_size), trt.nptype(b_type)
            )
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            hdm_obj = HostDeviceObj(host_mem, device_mem, tuple(b_size))
            if engine.binding_is_input(binding):
                inputs.append(hdm_obj)
            else:
                outputs.append(hdm_obj)
        return inputs, outputs, bindings

    def set_inputs(self, idx: int, img: np.ndarray) -> None:
        self.inputs[idx].host = img

    @staticmethod
    def _process_outputs_(outputs: List[Any]) -> List[np.ndarray]:
        preds = list()
        for output in outputs:
            pred = np.asarray(output.host)
            pred = pred.reshape(output.size)
            preds.append(pred)
        return preds

    # noinspection PyArgumentList
    def run(self, img: np.ndarray) -> List[np.ndarray]:
        self.set_inputs(0, img)
        with self.engine.create_execution_context() as context:
            for _input in self.inputs:
                cuda.memcpy_htod_async(_input.device, _input.host, self.stream)

            context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle,
            )

            for _output in self.outputs:
                cuda.memcpy_dtoh_async(
                    _output.host, _output.device, self.stream
                )

            self.stream.synchronize()

        preds = self._process_outputs_(self.outputs)
        return preds


class TRTWrapper:
    def __init__(
        self,
        onnx_path: Optional[str] = None,
        trt_path: Optional[str] = None,
        gpu_device_id: Optional[int] = None,
        intra_op_num_threads: Optional[int] = 0,
        inter_op_num_threads: Optional[int] = 0,
        fp16_mode: bool = False,
    ) -> None:
        """
        :param onnx_path: onnx-file path, default = None
        :param trt_path: onnx-file path, default = None
        :param gpu_device_id: gpu device id to use, default = 0
        :param intra_op_num_threads: ort_session_options.intra_op_num_threads,
            to let onnxruntime choose by itself is required 0, default = 0
        :param inter_op_num_threads: ort_session_options.inter_op_num_threads,
            to let onnxruntime choose by itself is required 0, default = 0
        :param fp16_mode: use fp16_mode if class initializes only with
            onnx_path on GPU, default = False
        :type onnx_path: str
        :type trt_path: str
        :type gpu_device_id: int
        :type intra_op_num_threads: int
        :type inter_op_num_threads: int
        :type fp16_mode: bool
        """
        if gpu_device_id is None:
            import onnxruntime

            self.is_using_tensorrt = False
            ort_session_options = onnxruntime.SessionOptions()
            ort_session_options.intra_op_num_threads = intra_op_num_threads
            ort_session_options.inter_op_num_threads = inter_op_num_threads
            self.ort_session = onnxruntime.InferenceSession(
                onnx_path, ort_session_options
            )

        else:
            self.is_using_tensorrt = True
            if trt_path is None:
                builder = TRTEngineBuilder()
                trt_path = builder.build_engine(onnx_path, fp16_mode=fp16_mode)

            self.trt_session = TRTRunWrapper(trt_path)

    def run(self, img: Any) -> Any:
        """
        run example:
        img = self._process_img_(img)
        if self.is_using_tensorrt:
            preds = self.trt_session.run(img)
        else:
            ort_inputs = {self.ort_session.get_inputs()[0].name: img}
            preds = self.ort_session.run(None, ort_inputs)

        preds = self._process_preds_(preds)
        """
        pass
