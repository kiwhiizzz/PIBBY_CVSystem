import os
from datetime import datetime

from ultralytics import YOLO

from src.utils.config import EPOCHS, BATCH_SIZE, IMG_SIZE


def main():
    model = YOLO("yolov8n-cls.pt")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo_cls_{timestamp}"

    results = model.train(
        data="data/processed",
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        name=run_name,
        exist_ok=False,
    )

    best_path = str(model.trainer.best)
    top1 = results.top1

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    os.makedirs("models", exist_ok=True)
    with open("models/registro.txt", "a", encoding="utf-8") as f:
        f.write(
            f"YOLOv8-cls | {best_path} | Epochs: {EPOCHS} | batch: {BATCH_SIZE} | "
            f"exactitud: {top1:.1%} | date: {fecha}\n"
        )

    print("Resultado agregado al registro")


if __name__ == "__main__":
    main()