# from insightface.app import FaceAnalysis    
# from app import FaceAnalysis    
from pathlib import Path
from model_zoo import get_model

INSIGHTFACE_ROOT = Path('~/.insightface').expanduser()
INSIGHT_MODELS = INSIGHTFACE_ROOT / "models"
model_zoo = ['buffalo_l', 'buffalo_m', 'buffalo_s']
model_pack_name = model_zoo[1]

# Build the path to the TRT engine.
model_file = INSIGHT_MODELS / model_pack_name / "det_10g.trt"
# Make sure you have the corresponding ONNX file as well for routing.
onnx_file = INSIGHT_MODELS / model_pack_name / "det_10g.onnx"

# Pass the TRT file path via kwargs (or adjust get_model accordingly)
analy_app = get_model(str(model_pack_name), trt_file=str(model_file))
analy_app.prepare(ctx_id=0, det_size=(640, 640))

def detect_faces(img):
    faces = analy_app.get(img, max_num=0)
    return faces
