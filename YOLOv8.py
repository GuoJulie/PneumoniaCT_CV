"""
data:
    CT         :     used
    mask       : not used
    labels(txt):     used
    labelsJson : not used
    dataset.yaml:    used
"""

from ultralytics import YOLO
import os
import shutil
import torch
import multiprocessing


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU")
    else:
        device = torch.device("cpu")
        print("CPU")

    multiprocessing.freeze_support()
    torch.cuda.empty_cache()

    YOLODataset_path = "./YOLODataset"
    models_path = "./models/"

    save_result_path = "./runs"
    if os.path.exists(save_result_path):
        shutil.rmtree(save_result_path)

    # Train the model
    # Load a model
    yolov8_premodel_path = os.path.join(models_path, "yolov8n-seg.pt")
    yaml_dataset_path = os.path.join(YOLODataset_path, "dataset.yaml")
    model = YOLO(yolov8_premodel_path)  # load a pretrained model

    results = model.train(
        data=yaml_dataset_path,
        epochs=1,
        # imgsz=640,
        batch=2,
        flipud=0.5,
        # device="cpu",
        # device=0,
        seed=2024,
        amp=True,
    )

    model.val()

    model = YOLO("./runs/segment/train/weights/best.pt")

    source = "./test/testpng/"
    for f in os.listdir(source):
        file_path = os.path.join(source, f)
        model.predict(file_path, save=True, retina_masks=True)
