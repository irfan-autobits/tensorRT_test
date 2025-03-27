# from insightface.app import FaceAnalysis    
# from app import FaceAnalysis    
from pathlib import Path
# from insightface.model_zoo import get_model
from model_zoo import get_model


INSIGHTFACE_ROOT = Path('~/.insightface').expanduser()
INSIGHT_MODELS = INSIGHTFACE_ROOT / "models"
model_zoo = ['buffalo_l', 'buffalo_m', 'buffalo_s']
model_pack_name = model_zoo[1]

model_file = INSIGHT_MODELS / model_pack_name / "det_10g.trt"
analy_app = get_model(str(model_file))
analy_app.prepare(ctx_id=0, det_size=(640, 640))

def detect_faces(img):
    faces = analy_app.get(img, max_num=0)  # Runs both detection and recognition by default
    return faces
