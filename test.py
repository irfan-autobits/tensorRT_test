# from insightface.app import FaceAnalysis    
# from app import FaceAnalysis    
from pathlib import Path
from model_zoo import get_model

# Define model paths.
INSIGHTFACE_ROOT = Path('~/.insightface').expanduser()
INSIGHT_MODELS = INSIGHTFACE_ROOT / "models"
model_zoo = ['buffalo_l', 'buffalo_m', 'buffalo_s']
model_pack_name = model_zoo[1]

# Paths for TRT and ONNX files.
trt_file = INSIGHT_MODELS / model_pack_name / "det_10g.trt"
onnx_file = INSIGHT_MODELS / model_pack_name / "det_10g.onnx"

# Load the detection model using get_model.
analy_app = get_model(str(model_pack_name), trt_file=str(trt_file))
if analy_app is None:
    raise RuntimeError("Failed to load detection model.")

# Prepare the model.
analy_app.prepare(ctx_id=0, det_size=(640, 640))

def detect_faces(img):
    faces = analy_app.get(img, max_num=0)
    return faces

if __name__ == '__main__':
    import cv2
    img = cv2.imread("your_image.jpg")  # Replace with a valid image file.
    det, kps = analy_app.detect(img)
    print("Detection results:", det)
