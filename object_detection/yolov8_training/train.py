from ultralytics import YOLO
import torch

def main():
    model = YOLO("runs/detect/train5/weights/best.pt")  #runs/detect/trainX/weights/best.pt
    model.train(
        data="datasets/myproject-7/data.yaml",
        imgsz=640,
        epochs=100,
        batch=8,      # 4GB VRAM safer than 16
        device=0,
        workers=4     # now safe
    )

if __name__ == "__main__":
    main()

