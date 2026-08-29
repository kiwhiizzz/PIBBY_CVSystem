from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def detect_face(frame):
    results = model(frame, verbose = False)
    boxes =  results[0].boxes

    if len(boxes) == 0:
        return None

    best_box = boxes[boxes.conf.argmax()]
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    confidence = best_box.conf.item()

    return {
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
        "confidence" : confidence
    }