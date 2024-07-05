from ultralytics import YOLO

#model = YOLO("yolov8x-seg.pt")
model = YOLO("../yolo/runs/segment/patches_yolov8x_seg_500e/weights/last.pt")

results = model.train(
        batch=16,
        data="dataset_seg.yaml",
	device=[0],
        epochs=300,
        save=True,
	name='patches_yolov8x_seg_800e',
    )
