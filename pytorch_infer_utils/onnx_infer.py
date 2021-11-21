from typing import Any, Optional


class ONNXWrapper:
    def __init__(
        self,
        onnx_path: str,
        gpu_device_id: Optional[int] = None,
        intra_op_num_threads: Optional[int] = 0,
        inter_op_num_threads: Optional[int] = 0,
    ) -> None:
        """
        :param onnx_path: onnx-file path, required
        :param gpu_device_id: gpu device id to use, default = None
        :param intra_op_num_threads: ort_session_options.intra_op_num_threads,
            to let onnxruntime choose by itself is required 0, default = 0
        :param inter_op_num_threads: ort_session_options.inter_op_num_threads,
            to let onnxruntime choose by itself is required 0, default = 0
        :type onnx_path: str
        :type gpu_device_id: int
        :type intra_op_num_threads: int
        :type inter_op_num_threads: int
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
            import onnx
            import onnx_tensorrt.backend as backend

            self.is_using_tensorrt = True
            model_proto = onnx.load(onnx_path)
            for gr_input in model_proto.graph.input:
                gr_input.type.tensor_type.shape.dim[0].dim_value = 1

            self.engine = backend.prepare(
                model_proto, device=f"CUDA:{gpu_device_id}"
            )

    def run(self, img: Any) -> Any:
        """
        run example:
        img = self._process_img_(img)
        if self.is_using_tensorrt:
            preds = self.engine.run(img)
        else:
            ort_inputs = {self.ort_session.get_inputs()[0].name: img}
            preds = self.ort_session.run(None, ort_inputs)

        preds = self._process_preds_(preds)
        """
        pass
