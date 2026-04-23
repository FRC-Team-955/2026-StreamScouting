from ultralytics import YOLO

model = YOLO("yolov8c.pt")

model.train(
    data="dataset2/data.yaml",
    epochs=50,
    imgsz=640,
    device=0  #device=0 for CUDA, device='mps' for apple silicon, device='cpu' if youre a bum. remove this line for auto detect
)
