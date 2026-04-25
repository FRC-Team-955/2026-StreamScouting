from ultralytics import YOLO

def main():
    model = YOLO("last.pt") # change to best
    model.train(
    data="dataset2/data.yaml",
    epochs=190,
    imgsz=640,
    device=0,
    workers=10,
    batch=28,
    hsv_h=0.015,   # hue shift — different lighting
    hsv_s=0.7,     # saturation — washed out/dark venues  
    hsv_v=0.4,     # brightness — overexposed/dark
    scale=0.9,     # zoom out — makes robots appear smaller
    mosaic=1.0,    # already on by default, keeps it
    fliplr=0.5,    # horizontal flip
)
if __name__ == "__main__":
    main()
