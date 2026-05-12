from ai.yolo_engine.model import YoloModel

def run_yolo(frame):
    model = YoloModel.get_instance()
    if not model:
        return {"annotated_frame": frame, "detections": []}

    results = model(frame, verbose=False)
    detections = []
    annotated_frame = frame.copy()

    for r in results:
        annotated_frame = r.plot()
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            detections.append({
                "label": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return {"annotated_frame": annotated_frame, "detections": detections}
