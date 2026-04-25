from ultralytics import YOLO
# this is tuned for my system. Turn up batch until GPU memory full, turn up workers until RAM full
# add cache=True if you have a lot of RAM
def main():
    model = YOLO("yolov8m.pt") # change to best
    model.train(
    data="dataset2/data.yaml",
    epochs=190,
    imgsz=640,
    device=0,
    workers=10,
    batch=28,
)
if __name__ == "__main__":
    main()
