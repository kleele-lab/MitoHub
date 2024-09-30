from ultralytics import YOLO

model_path = 'runs/segment/yolov8x_300e_1280/weights/last.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

# Predict with the model
results = model("segment_test.png", imgsz = 1280,
                retina_masks=True, overlap_mask=True, save=True, show_labels=False, line_width=0, show_boxes=False)
