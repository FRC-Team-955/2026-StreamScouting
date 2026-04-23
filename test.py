from ultralytics import YOLO

model = YOLO("yolov8c.pt")

results = model("dataset/test/images", save=True)