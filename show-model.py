from pathlib import Path
from ultralytics import YOLO

PATH=Path("models/reduced_min_320_tiny.yaml")

model = YOLO(PATH)

model.info(verbose=True)
