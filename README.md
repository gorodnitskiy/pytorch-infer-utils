
# PyTorch Infer Utils

This package proposes simplified exporting pytorch models to ONNX and TensorRT,
and also gives some base interface for model inference.

## To install
```shell
git clone https://github.com/gorodnitskiy/pytorch_infer_utils.git
pip install /path/to/pytorch_infer_utils/
```

## Export PyTorch model to ONNX
- Check model for denormal weights to achieve better performance.
  Use ```load_weights_rounded_model``` func to load model with weights rounding:
  ```
  from pytorch_infer_utils import load_weights_rounded_model

  model = ModelClass()
  load_weights_rounded_model(
      model,
      "/path/to/model_state_dict",
      map_location=map_location
  )
  ```
- Use ```ONNXExporter.torch2onnx``` method to export pytorch model to ONNX:
  ```
  from pytorch_infer_utils import ONNXExporter

  model = ModelClass()
  model.load_state_dict(
      torch.load("/path/to/model_state_dict", map_location=map_location)
  )
  model.eval()

  exporter = ONNXExporter()
  input_shapes = [-1, 3, 224, 224] # -1 means that is dynamic shape
  exporter.torch2onnx(model, "/path/to/model.onnx", input_shapes)
  ```
- Use ```ONNXExporter.optimize_onnx``` method to optimize ONNX
  via [onnxoptimizer](https://github.com/onnx/optimizer):
  ```
  from pytorch_infer_utils import ONNXExporter

  exporter = ONNXExporter()
  exporter.optimize_onnx("/path/to/model.onnx", "/path/to/optimized_model.onnx")
  ```
- Use ```ONNXExporter.optimize_onnx_sim``` method to optimize ONNX via
  [onnx-simplifier](https://github.com/daquexian/onnx-simplifier).
  Be careful with onnx-simplifier not to lose dynamic shapes.
  ```
  from pytorch_infer_utils import ONNXExporter

  exporter = ONNXExporter()
  exporter.optimize_onnx_sim("/path/to/model.onnx", "/path/to/optimized_model.onnx")
  ```
- Also, a method combined the above methods is available
```ONNXExporter.torch2optimized_onnx```:
  ```
  from pytorch_infer_utils import ONNXExporter

  model = ModelClass()
  model.load_state_dict(
      torch.load("/path/to/model_state_dict", map_location=map_location)
  )
  model.eval()

  exporter = ONNXExporter()
  input_shapes = [1, 3, 224, 224]
  exporter.torch2optimized_onnx(model, "/path/to/model.onnx", input_shapes)
  ```
- Other params that can be used in class initialization:
  - default_shapes: default shapes if dimension is dynamic,
default = [1, 3, 224, 224]
  - onnx_export_params:
    - export_params: store the trained parameter weights inside the model file,
default = True
    - do_constant_folding: whether to execute constant folding for
optimization, default = True
    - input_names: the model's input names, default = ["input"]
    - output_names: the model's output names, default = ["output"]
    - opset_version: the ONNX version to export the model to, default = 11
  - onnx_optimize_params:
    - fixed_point: use fixed point, default = False
    - passes: optimization passes, default = [
      "eliminate_deadend", "eliminate_duplicate_initializer",
      "eliminate_identity", "eliminate_if_with_const_cond",
      "eliminate_nop_cast", "eliminate_nop_dropout",
      "eliminate_nop_flatten", "eliminate_nop_monotone_argmax",
      "eliminate_nop_pad", "eliminate_nop_transpose",
      "eliminate_unused_initializer", "extract_constant_to_initializer",
      "fuse_add_bias_into_conv", "fuse_bn_into_conv",
      "fuse_consecutive_concats", "fuse_consecutive_log_softmax",
      "fuse_consecutive_reduce_unsqueeze", "fuse_consecutive_squeezes",
      "fuse_consecutive_transposes", "fuse_matmul_add_bias_into_gemm",
      "fuse_pad_into_conv", "fuse_transpose_into_gemm",
      "lift_lexical_references", "nop",
    ]

## Export ONNX to TensorRT
- Check TensorRT health via ```check_tensorrt_health``` func
- Use ```TRTEngineBuilder.build_engine``` method to export ONNX to TensorRT:
  ```
  from pytorch_infer_utils import TRTEngineBuilder

  exporter = TRTEngineBuilder()
  # get engine by itself
  engine = exporter.build_engine("/path/to/model.onnx")
  # or save engine to /path/to/model.trt
  exporter.build_engine("/path/to/model.onnx", engine_path="/path/to/model.trt")
  ```
- fp16_mode is available:
  ```
  from pytorch_infer_utils import TRTEngineBuilder

  exporter = TRTEngineBuilder()
  engine = exporter.build_engine("/path/to/model.onnx", fp16_mode=True)
  ```
- int8_mode is available. It requires calibration_set of items as
```List[Any]```, ```load_item_func``` - func to correctly read and process
item (image), ```max_item_shape``` - max item size as [C, H, W] to allocate correct
size of memory:
  ```
  from pytorch_infer_utils import TRTEngineBuilder

  exporter = TRTEngineBuilder()
  engine = exporter.build_engine(
      "/path/to/model.onnx",
      int8_mode=True,
      calibration_set=calibration_set,
      max_item_shape=max_item_shape,
      load_item_func=load_item_func,
  )
  ```
- Also, additional params for builder config
```builder.create_builder_config``` can be put to kwargs.
- Other params that can be used in class initialization:
  - use_opt_shapes: use optimal shapes config option, default = False
  - opt_shape_dict: optimal shapes,
{'input_name': [minimal_shapes, average_shapes, maximal_shapes]}, all shapes
required as [B, C, H, W], default = {'input': [[1, 3, 224, 224],
[1, 3, 224, 224], [1, 3, 224, 224]]}
  - max_workspace_size: max workspace size, default = [1, 30]
  - stream_batch_size: batch size for forward network during transferring to
int8, default = 100
  - cache_file: int8_mode cache filename, default = "model.trt.int8calibration"

## Inference via onnxruntime on CPU and onnx_tensort on GPU
- Base class ONNXWrapper ```__init__``` has the structure as below:
  ```
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
  ```
- ```ONNXWrapper.run``` method assumes the use of such a structure:
  ```
  img = self._process_img_(img)
  if self.is_using_tensorrt:
      preds = self.engine.run(img)
  else:
      ort_inputs = {self.ort_session.get_inputs()[0].name: img}
      preds = self.ort_session.run(None, ort_inputs)

  preds = self._process_preds_(preds)
  ```

## Inference via onnxruntime on CPU and TensorRT on GPU
- Base class TRTWrapper ```__init__``` has the structure as below:
  ```
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
      :param gpu_device_id: gpu device id to use, default = None
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
  ```
- ```TRTWrapper.run``` method assumes the use of such a structure:
  ```
  img = self._process_img_(img)
  if self.is_using_tensorrt:
      preds = self.trt_session.run(img)
  else:
      ort_inputs = {self.ort_session.get_inputs()[0].name: img}
      preds = self.ort_session.run(None, ort_inputs)

  preds = self._process_preds_(preds)
  ```

## Environment

### Docker
- Use nvcr.io/nvidia/tensorrt:21.05-py3 image due to
[issue](https://github.com/NVIDIA-AI-IOT/torch2trt/issues/557)
with max_workspace_size attribute in TensorRT 8.0.0.3.
- This image has already contained all CUDA required dependencies,
including additional python packages.
```shell
cd /path/to/pytorch_infer_utils/
docker build --tag piu .
docker run \
    --rm \
    -it \
    --user $(id -u):$(id -g) \
    --volume </path/to/target_folder>:/workspace:rw \
    --name piu_test \
    --gpus '"device=0"' \
    --entrypoint /bin/bash/ \
    piu
```

### Manual installation

#### TensorRT
- TensorRT installation guide is
[here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
- Required CUDA-Runtime, CUDA-ToolKit
- Also, required additional python packages not included to ```setup.cfg```
(it depends upon CUDA environment version):
  - pycuda
  - nvidia-tensorrt
  - nvidia-pyindex

#### onnx_tensorrt
- [onnx_tensorrt](https://github.com/onnx/onnx-tensorrt) requires CUDA-Runtime
and TensorRT.
- Use git clone for onnx-tensorrt installation due to
[issue](https://github.com/onnx/onnx-tensorrt/blob/e9456d57605c883cdf985e634ab483e2c1500bb1/setup.py#L5)
with import onnx_tensorrt in onnx_tensorrt setup.py:
  ```shell
  git clone https://github.com/onnx/onnx-tensorrt.git
  cd onnx-tensorrt
  cp -r onnx_tensorrt /usr/local/lib/python3.8/dist-packages
  cd ..
  rm -rf onnx-tensorrt
  ```
