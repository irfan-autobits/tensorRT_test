import os
import os.path as osp
import glob
import onnxruntime
from .retinaface import *

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initialize CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class PickableInferenceSession(onnxruntime.InferenceSession): 
    # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        model_path = values['model_path']
        self.__init__(model_path)

class ModelRouter:
    def __init__(self, onnx_file, trt_file):
        self.onnx_file = onnx_file
        self.trt_file = trt_file

    def get_model(self, **kwargs):
        # Create an ONNX session for model inspection.
        session = PickableInferenceSession(self.onnx_file, **kwargs)
        print(f'Applied providers: {session._providers}, with options: {session._provider_options}')
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()
        # For example, if outputs>=5 assume a detection model.
        if len(outputs) >= 5:
            # Set use_onnx=False so RetinaFace will use TRT.
            return RetinaFace(model_file=self.onnx_file,
                              session=session,
                              use_onnx=False,
                              trt_file=self.trt_file)
        else:
            #raise RuntimeError('error on model routing')
            return None

def find_model_file(dir_path, extension):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob(f"{dir_path}/*.{extension}")
    if len(paths) == 0:
        return None
    return sorted(paths)[-1]

def get_default_providers():
    return ['CUDAExecutionProvider', 'CPUExecutionProvider']

def get_default_provider_options():
    return None

def get_model(name, **kwargs):
    root = kwargs.get('root', '~/.insightface')
    root = os.path.expanduser(root)
    model_root = osp.join(root, 'models')
    allow_download = kwargs.get('download', False)
    download_zip = kwargs.get('download_zip', False)
    
    # If name is not an absolute filename, treat it as a model pack name.
    if not name.endswith('.onnx'):
        model_dir = osp.join(model_root, name)
        onnx_file = find_model_file(model_dir, "onnx")
        trt_file = find_model_file(model_dir, "trt")
        if onnx_file is None or trt_file is None:
            print("Required model files not found.")
            return None
    else:
        onnx_file = name
        # For this example, assume the corresponding TRT file is provided via kwargs.
        trt_file = kwargs.get('trt_file')
    
    # Ensure files exist.
    assert osp.exists(onnx_file), f'ONNX model file {onnx_file} should exist'
    assert osp.isfile(onnx_file), f'ONNX model file {onnx_file} should be a file'
    assert osp.exists(trt_file), f'TRT engine file {trt_file} should exist'
    assert osp.isfile(trt_file), f'TRT engine file {trt_file} should be a file'
    
    router = ModelRouter(onnx_file, trt_file)
    providers = kwargs.get('providers', get_default_providers())
    provider_options = kwargs.get('provider_options', get_default_provider_options())
    model = router.get_model(providers=providers, provider_options=provider_options)
    return model