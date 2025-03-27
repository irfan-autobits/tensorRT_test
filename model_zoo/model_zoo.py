# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

import os
import os.path as osp
import glob
import onnxruntime
# from .arcface_onnx import *
from .retinaface import *
#from .scrfd import *
# from .landmark import *
# from .attribute import Attribute
# from .inswapper import INSwapper
# from ..utils import download_onnx

__all__ = ['get_model']


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
        session = PickableInferenceSession(self.onnx_file, **kwargs)
        print(f'Applied providers: {session._providers}, with options: {session._provider_options}')
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()

        # If the model has 5 or more outputs, assume it's a detection model.
        if len(outputs) >= 5:
            # Here we set use_onnx=False so RetinaFace will use TRT mode.
            return RetinaFace(model_file=self.onnx_file, session=session, use_onnx=False, trt_file=self.trt_file)
        # elif input_shape[2]==192 and input_shape[3]==192:
        #     return Landmark(model_file=self.onnx_file, session=session)
        # elif input_shape[2]==96 and input_shape[3]==96:
        #     return Attribute(model_file=self.onnx_file, session=session)
        # elif len(inputs)==2 and input_shape[2]==128 and input_shape[3]==128:
        #     return INSwapper(model_file=self.onnx_file, session=session)
        # elif input_shape[2]==input_shape[3] and input_shape[2]>=112 and input_shape[2]%16==0:
        #     return ArcFaceONNX(model_file=self.onnx_file, session=session)
        else:
            #raise RuntimeError('error on model routing')
            return None

def find_onnx_file(dir_path):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob("%s/*.onnx" % dir_path)
    paths = glob.glob("%s/*.trt" % dir_path)
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]

def find_model_file(dir_path, extension):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob(f"{dir_path}/*.{extension}")
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]

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
    
    # If the given name is not an absolute file, build the directory path.
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