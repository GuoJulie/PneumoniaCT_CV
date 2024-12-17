"""
# ===============Part_2: Dataset Production===============

To do: <Strategy>
    1. Read images/labels/labelsJSPN'path list
        Input:
            images[] (.png)
            labelsJSON[] (.json)
            labels[] (.txt)
    2. Create YOLO dataset: 
        [1] split set & copy files
        [2] create config file (.yaml) 
        Structure:
                dataset:(dir: sub_dir)
                        train/val/test: images
                                        labels -> YOLO TXT (annotation file)
                                        labelsJson
                dataset.yaml (config file)
                        -> format:
                                train: trainset_absolute_path
                                val: valset_absolute_path
                                # test: testset_absolute_path
                                nc: num(classes)
                                names: ['names_class', ...]
"""

import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # progress bar
import shutil

MODELTYPES = ["YOLO", "UNET"]


def make_dataset(
    model, data_path, train_size=0.8, test_set=True
):

    # create dataset folder
    global MODELTYPES
    if model in MODELTYPES:
        dataset_path = os.path.join(os.path.dirname(__file__), f"{str(model)}Dataset")
        if model == "YOLO":
            yaml_path = os.path.join(dataset_path, "dataset.yaml")
    else:
        print(
            f"Missing creating {str(model)}'s Dataset solution. Recommended: {', '.join(MODELTYPES)}"
        )
        return
    ensure_dir(dataset_path)

    images_path = os.path.join(data_path, "images")
    masks_path = os.path.join(data_path, "masks")
    labels_path = os.path.join(data_path, "labels")
    labelsJSON_path = os.path.join(data_path, "labelsJSON")

    # 1. Read images/labels/labelsJSPN'path list
    images = []
    masks = []
    labels = []
    labelsJSON = []
    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path, image_name)
        mask_path = os.path.join(masks_path, image_name)

        name, ext = os.path.splitext(image_name)
        label_name = os.path.join(name + ".txt")
        label_path = os.path.join(labels_path, label_name)

        labelJSON_name = os.path.join(name + ".json")
        labelJSON_path = os.path.join(labelsJSON_path, labelJSON_name)

        if all(map(os.path.exists, [mask_path, label_path, labelJSON_path])):
            images.append(image_path)
            masks.append(mask_path)
            labels.append(label_path)
            labelsJSON.append(labelJSON_path)
        else:
            print("The initial data in images/masks/labels/labelsJSON do not correspond")

    # 2.1.1 Split data to train/val/test set
    data = list(zip(images, masks, labels, labelsJSON))

    if test_set:
        test_size = int(len(images) * (1 - train_size) / 2)
        train_val_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )  # split
        train_data, val_data = train_test_split(
            train_val_data, test_size=test_size, random_state=42
        )  # test_size: specific len(val_images)
    else:
        train_data, val_data = train_test_split(
            data, test_size=1 - train_size, random_state=42
        )  # split # test_size: radio
        test_data = list(), list(), list(), list()

    train_images, train_masks, train_labels, train_labelsJSON = zip(*train_data)  # unpack
    val_images, val_masks, val_labels, val_labelsJSON = zip(*val_data)
    test_images, test_masks, test_labels, test_labelsJSON = (zip(*test_data) if test_set else ([], [], [], []))

    # make CSV ???

    # dataset_structure
    dataset_structure = {
        "train": {
            "images": train_images,
            "masks": train_masks,
            "labels": train_labels,
            "labelsJson": train_labelsJSON,
        },
        "val": {
            "images": val_images,
            "masks": val_masks,
            "labels": val_labels,
            "labelsJson": val_labelsJSON,
        },
        "test": {
            "images": test_images,
            "masks": test_masks,
            "labels": test_labels,
            "labelsJson": test_labelsJSON,
        },
    }

    # 2.1.2 Copy data to train/val/test set
    dataset_paths = {}
    for base_dir, sub_dirs in dataset_structure.items():
        dataset_paths[base_dir] = {}
        for sub_dir, file_list in sub_dirs.items():
            # create each sub_dir & save paths
            sub_dir_path = os.path.join(dataset_path, base_dir, sub_dir)
            dataset_paths[base_dir][sub_dir] = sub_dir_path
            if base_dir == "test" and test_set is False:
                continue
            if sub_dir == "masks" and model == "YOLO":
                continue
            if (sub_dir == "labels" or sub_dir == "labelsJson") and model == "UNET":
                continue

            ensure_dir(sub_dir_path)

            # copy
            # print(f"\nlength: {base_dir + '_' + sub_dir}: {len(file_list)}")
            copy_files(sub_dir_path, file_list)

    # 2.2 Create config file (.yaml)
    if model == "YOLO":
        """ YAML
        format:
            train: trainset_absolute_path
            val: valset_absolute_path
            # test: testset_absolute_path
            nc: num(classes)
            names: ['names_class', ...]
        """
        class_mapping = {"pneumonia": 1}
        YAML_OUT = {
            "train": os.path.abspath(dataset_paths["train"]["images"]),
            "val": os.path.abspath(dataset_paths["val"]["images"]),
            'test': os.path.abspath(dataset_paths["test"]["images"]),
            "nc": len(class_mapping),
            "names": list(class_mapping.keys())
        }

        with open(yaml_path, "w") as yaml_output:
            idx = 0
            for key, value in YAML_OUT.items():
                idx += 1
                if key == 'test' and test_set is False:
                    continue
                row = f'{key}: {value}'
                if idx != len(YAML_OUT):
                    row = row + '\n'
                yaml_output.write(row)

    print(f"\n{str(model)}Dataset was successfully created.")

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def copy_files(sub_dir_path, file_list):
    # copy files to the dataset folders
    for file_path in tqdm(file_list, desc=sub_dir_path):
        shutil.copy(
            file_path,
            os.path.join(sub_dir_path, os.path.basename(file_path)),
        )


data_path = "./data/mosmed/data_preprocessing/"
# make_dataset("YOLO", data_path, train_size=0.8, test_set=False)
make_dataset("UNET", data_path, train_size=0.9, test_set=False)
