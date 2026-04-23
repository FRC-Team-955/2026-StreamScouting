from ultralytics import YOLO

model = YOLO("yolov8c.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    device='cpu'  #device=0 for CUDA, device='mps' for apple silicon. remove this line for auto detect
)
