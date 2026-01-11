import cv2
import time
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from model_utils import load_classifier, SquarePad

# Config
VIDEO_PATH = 'inputVideo/traffic_test.mp4' 
if not os.path.exists('outputVideo'):
    os.makedirs('outputVideo')
OUTPUT_PATH = 'outputVideo/output_video.mp4'
YOLO_MODEL_PATH = "models/best_yolo.pt"
CLASSIFIER_PATH = 'models/best_classifier.pth'
CLASS_NAMES_PATH = 'models/class_names.txt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 165, 0),  # Cyan
    (128, 0, 128),  # Purple
    (0, 165, 255)   # Orange
]

preprocess = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = []
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    print(f"Class names file not found at {CLASS_NAMES_PATH}")
    class_names = ["Unknown"] * 10 

print(f"Classes: {class_names}")

def process_video():
    if not os.path.exists(YOLO_MODEL_PATH):
         print(f"YOLO model not found at {YOLO_MODEL_PATH}, Ultralytics will attempt to download 'yolov8n.pt' or error out.")
    
    detector = YOLO(YOLO_MODEL_PATH)
    
    try:
        classifier = load_classifier(CLASSIFIER_PATH, len(class_names))
    except Exception as e:
        print(f"Could not load classifier: {e}")
        classifier = None

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found at {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"Processing video... ({width}x{height} @ {fps}fps)")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        t0 = time.time()
        results = detector(frame, verbose=False, conf=0.25)
        yolo_time = time.time() - t0

        batch_tensors = []
        batch_coords = [] 

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                w_box = x2 - x1
                h_box = y2 - y1
                if w_box < 50 or h_box < 50: continue
                if x1 >= x2 or y1 >= y2: continue

                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size == 0: continue
                
                batch_coords.append((x1, y1, x2, y2))

                if classifier:
                    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    batch_tensors.append(preprocess(pil_img))

        pred_labels = []
        pred_scores = []
        pred_indices = []
        cls_time = 0.0

        if classifier and batch_tensors:
            input_batch = torch.stack(batch_tensors).to(DEVICE)

            t1 = time.time()
            with torch.no_grad():
                outputs = classifier(input_batch) 
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf_cls, pred_idx_tensor = torch.max(probs, 1)
                
                pred_indices = pred_idx_tensor.cpu().tolist()
                pred_scores = conf_cls.cpu().tolist()
            cls_time = time.time() - t1
        
        elif not classifier: 
             pred_indices = [0] * len(batch_coords)
             pred_scores = [0.0] * len(batch_coords)
             cls_time = 0.0

        if frame_count % 10 == 0: 
            print(f"   Frame {frame_count}... Detector: {yolo_time:.4f}s | Classifier: {cls_time:.4f}s | Cars Detected: {len(batch_coords)}")

        for i, (x1, y1, x2, y2) in enumerate(batch_coords):
            if i < len(pred_indices):
                idx = pred_indices[i]
                score = pred_scores[i]
                
                label = class_names[idx] if idx < len(class_names) else "Unknown"
                color = COLORS[idx % len(COLORS)]
            else:
                label = "Vehicle"
                score = 0.0
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {score:.0%}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\nVideo saved at: {OUTPUT_PATH}")

# EXECUTE
try:
    process_video()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
