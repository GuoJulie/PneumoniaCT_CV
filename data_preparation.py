"""
# ===============Part_1: Original Data Format Conversion===============

To do: <Strategy>
    1. init data [.nii(Nifit file) -> .png]: 
            CT data: 3D -> 2D
            save: [1] .png (dir: data_preprocessing/images)
    2. mask data [.nii -> .json / .txt]: 
            CT mask data: 3D -> 2D 
                          Filter the masks(contains 'pneumonia') -> JSON & YOLO_TXT 
            save: [1] .png (dir: data_preprocessing/masks)
                  [2] .json (dir: data_preprocessing/labelJSON) 
                            (recognizable by Labelme.exe)
                  [3] .txt(dir: data_preprocessing/labels)
                            (for YOLO input: 
                                YOLO format annotation data file -> describe label's classID & relative position) 
    3. data augmentation:
            [1] CT (images) - optimize:
                [1.1] reduce the CT range and to increase the contrast (pkg: windowing -> transform window width & center)
                [1.2] Filter all CT & masks which contains 'pneumonia'
            [2] Masks - optimize: (for YOLOv8)
                [2.1] binarize masks (0 & 1)
                [2.2] padded & filled mask_tmp's connected area
                [2.3] close all contours in mask
                [2.4] contours -> polygons
                [2.5] discard the polygons(contours) if it contains few points
            [3] CT & Masks - optimize: -> create a train_data_generator to perform various transformations (for UNET)
                [3.1] define an image_generator -> keras.preprocessing.Image.ImageDataGenerator()
                [3.2] image data augmentation -> flow_from_directory()
                [3.3] image normalization (??? false -> binarize CT in real)
                    recommended: 
                        in U-Net : CT -> normalization is better than binarization
"""

import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import numpy as np
import io
import base64
from skimage import measure
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
)  # dilation & filling for mask
from PIL import Image  # Pillow Imaging Library
import json
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================1. Read Data========================
# Read .nii file
# def read_nifit(nii_path):
#     path_parts = nii_path.split("\\")
#     nii_name = path_parts[-1]
#     nii_type = path_parts[-2]

#     # read nii
#     img_slices = sitk.ReadImage(nii_path)
#     wincenter = img_slices
#     data_img_slices = sitk.GetArrayFromImage(img_slices)  # convert sitk image to numpy array

#     # write in txt
#     with open(f"{nii_type}_{nii_name.split('.')[0]}.txt", "w", encoding="utf-8") as f:
#         f.write(f"{nii_path}: \n {img_slices}\n")
#         f.write("===================================\n")
#         f.write(f"Shape of the data: {data_img_slices.shape}\n")
#         f.write(f"data_img_slices: \n")
#         for i in data_img_slices:
#             for j in i:
#                 f.write(f"{j}\n")
#             f.write("===================================\n")

#     return data_img_slices, nii_type, nii_name


# Make plt_savefig Name
def make_fig_name(nii_type, nii_name):
    return nii_type + "_" + nii_name.split(".")[0]


# Draw Nifit CT
def draw_image(data_img_slices, idx1, idx2, savefig=None):
    row = math.floor(math.sqrt(idx2 + 1 - idx1))
    col = math.ceil(math.sqrt(idx2 + 1 - idx1))
    if row * col < (idx2 + 1 - idx1):
        row += 1
    j = 1
    for i in range(idx1 - 1, idx2):
        plt.subplot(row, col, j)
        plt.imshow(data_img_slices[i], cmap="gray")
        plt.axis(False)
        j += 1
    plt.show()
    if savefig and isinstance(savefig, str):
        plt.savefig(f"CT_{savefig}.png")


# Draw CT histogram
def draw_histogram(data_img_slices, idx, savefig=None):
    ct_values = data_img_slices[idx - 1].flatten()  # flatten 3D -> 1D
    plt.figure(figsize=(10, 6))
    plt.hist(ct_values, bins=100, color="blue", alpha=0.7)
    plt.title("Histogram of CT Values")
    plt.xlabel("CT Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    if savefig and isinstance(savefig, str):
        plt.savefig(f"CT_histogram_{savefig}.png")


# Output nii's metadata
def create_metadata_dir(data_path):
    # create metadata folder & sub_dir
    meta_path = os.path.join(data_path, "metadata_tmp")
    img_meta_path = os.path.join(meta_path, "images")
    mask_meta_path = os.path.join(meta_path, "masks")
    PATH = {meta_path, img_meta_path, mask_meta_path}
    for path in PATH:
        os.makedirs(path, exist_ok=True)
    return img_meta_path, mask_meta_path


# Output nii's metadata
def write_metadata_in_txt(nii_path, sitk_nii, metadata, save_dir_path):
    meta_path = os.path.join(save_dir_path, f"{nii_name.split('.')[0]}_metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"{nii_path}: \n {sitk_nii}\n")
        f.write("===================================\n")
        f.write(f"Shape of the data: {metadata.shape}\n")
        f.write(f"data_img_slices: \n")
        for i in metadata:
            f.write("\n".join(map(str, i)))
            f.write("===================================\n")


# ========================2. Data Augmentation========================
# img
def window_transform(sitkImage, winwidth=250, wincenter=80):
    """
    To do: transform window width & center
           to reduce the CT range and to increase the contrast (pkg: windowing)

    :param sitkImage(input) -> SimpleITK object
    :param winwidth -> CT values range --> winwidth: smaller, contrast: stronger
    :param wincenter -> all displayed CT values' center position --> wincenter: smaller, brightness: brighter
    :return sitkImage(output) -> sitkImage after transform
    """
    # define borders
    min = int(wincenter - winwidth / 2.0)
    max = int(wincenter + winwidth / 2.0)

    # define SimpleITK object -> transform window width & center
    intensityWindow = sitk.IntensityWindowingImageFilter()
    intensityWindow.SetWindowMinimum(min)
    intensityWindow.SetWindowMaximum(max)
    # print(
    #     f"new window width: {intensityWindow.GetWindowMinimum()} ~ {intensityWindow.GetWindowMaximum()}"
    # )
    sitkImage = intensityWindow.Execute(sitkImage)

    return sitkImage

# img
def get_winwidth_wincenter(img_nii_path, mask_nii_path):
    """
    To do: According to CT's Mask flexibly winwidth && wincenter
    :param img_nii_path -> nii_path
    :param mask_nii_path -> nii_path
    :return winwidth, wincenter

    Note: for Lung CT: default winwidth: 1000 ~ 1600HU, default wincenter:-600 ~ -800HU
    """
    all_min_max_hu = []
    for nii_name in os.listdir(mask_nii_path):
        if nii_name.endswith(".nii.gz") or nii_name.endswith(".nii"):

            # read nii file
            imgs_nii = sitk.ReadImage(os.path.join(img_nii_path, nii_name))
            masks_nii = sitk.ReadImage(os.path.join(mask_nii_path, nii_name))

            # nii -> array[]
            imgs_init = sitk.GetArrayFromImage(imgs_nii)
            masks = sitk.GetArrayFromImage(masks_nii)

            masks_hu_init = np.where(masks == 1, imgs_init, np.nan)

            for mask in masks_hu_init:
                arr_mask = mask.flatten()
                if not np.all(np.isnan(arr_mask)):
                    all_min_max_hu.extend([np.nanmin(arr_mask), np.nanmax(arr_mask)])

    # print(f"all_min_max_hu: {all_min_max_hu}")

    lower_bound_min = (
        math.floor(np.nanmin(all_min_max_hu) / 100) * 100
    )  # find nearest bounds
    upper_bound_max = math.ceil(np.nanmax(all_min_max_hu) / 100) * 100
    winwidth = upper_bound_max - lower_bound_min
    wincenter = np.around(1 / 2 * (upper_bound_max + lower_bound_min))

    # print(f"min_bound: {min}, max_bound: {max}")
    # print(f"min: {lower_bound_min}, max: {upper_bound_max}")
    print(f"winwidth: {winwidth}, wincenter: {wincenter}")

    return winwidth, wincenter


# img
def img_to_base64str(img_pil):
    """
    To do:  Image convertion: PIL Image -> Base64 string

    :img_pil    : PIL Image Object(gray)
    :return     : (Base64)      str(utf-8)

    PIL: Python Imaging Library

    (Binary) byte str ==> use in Image/Video/...
    (Base64) byte str (ASCII character) ==> use in Print/Mail/URL/...
    (Base64)      str (utf-8: Unicode character (text str)) ==> use in JSON/XML/...

    This allows image data to be transmitted over the network as text strings or stored in text files without being affected by the compatibility or transmission issues that binary data can bring.
    """
    ENCODING = "utf-8"
    img_byte = io.BytesIO()  # Create a binary data stream in memory
    img_pil.save(img_byte, format="PNG")  # save img_pil in img_byte

    binary_byte_str = img_byte.getvalue()  # img -> Binary byte data
    base64_byte_str = base64.b64encode(
        binary_byte_str
    )  # (Binary) byte str -> (Base64) byte str
    base64_str = base64_byte_str.decode(ENCODING)  # (Base64) byte str -> (Base64) str

    return base64_str


# mask
def close_contour(contour):
    """
    To do: close contour [traite image shapes and borders]
           sure that a 2D contour is connected end-to-end to form a closed contour
    """
    if not np.array_equal(contour[0], contour[-1]):  # contours: not closed
        contour = np.vstack((contour, contour[0]))  # add contours[0] at the end
    return contour


# mask
def binary_mask_to_polygon(binary_mask, tolerance=0):
    """
    To do: binary_mask -> polygon(COCO dataset format)
    From project "pycococreator-master"

    :binary_mask : array[](uint8)
    :tolerance=0 : control the fineness of the polygon approximation process
    :return      : polygons (list)

    contours: {[c_1][c_2]...[c_N]} -> N contours in mask
    contour : [[y1,x1]...[yn,xn]]  -> n points in this contour
              -> shape: (n,2)
              -> y: row coordinate, x: col coordinate
    polygons: [[segmentation_1], ..., [segmentation_N]] -> N polygons in mask
    """
    # treat mask
    padded_binary_mask = np.pad(
        binary_mask, pad_width=1, mode="constant", constant_values=0
    )  # padding: to avoid losing the boundary info when check the contour later

    # check all contours in mask
    contours = measure.find_contours(
        padded_binary_mask, 0.5
    )  # threshold: 0.5 (for binary img) # default contour's precision = 1 -> decimals=1

    # treat contours
    polygons = []
    for contour in contours:
        contour = np.subtract(contour, 1)  # restore: original = contour - padding(1)

        contour = close_contour(contour)  # make sure contour closed
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:  # check points num # filter the 'contour' whose points < 3
            continue

        contour = np.flip(contour, axis=1)  # flip the axis: img(y,x) -> COCOdata(x,y)
        segmentation = contour.ravel().tolist()  # contour[points] -> 1D list
        segmentation = [
            0 if i < 0 else i for i in segmentation
        ]  # make sure segmentation >= 0
        polygons.append(segmentation)

    return polygons


# img & mask
def img_mask_to_json_txt(img, mask, class_names, class_mapping):
    """
    To do: img + mask -> json

    json: for img annotation tool (eg : Labelme)

    :img         : PIL Image Object (gray)
    :mask        : PIL Image Object (gray)
    :class_names : ['_background_', 'pneumonia']
    """
    # init JSON
    JSON_OUTPUT = {
        "version": "5.4.1",  # Labelme version
        "flags": {},  # Additional flag info
        "shapes": [],  # [{label, points, group_id, shape_type, flags}...{}]
        "imagePath": {},
        "imageData": {},  # Image data (Base64 encoded)
        "imageHeight": {},
        "imageWidth": {},
    }

    # init TXT -> YOLO .txt format: one polygon(object) <-> one row
    TXT_OUTPUT = []
    """
    TXT_OUTPUT = [
        # if bboxes is rectangle:
        [classID, x_center_nor, y_center_nor, width_nor, height_nor]
        ...  # here: nor-> normalized

        # for polygons in this case:
        [classID, [x1_nor, y1_nor], ...]
        ... # one polygon(object) <-> one row
    ] # list
    """

    # JSON - "imagePath" -> set externally

    # JSON - "imageData"
    imageData = img_to_base64str(img)  # Image -> base64_str ('utf-8')
    JSON_OUTPUT["imageData"] = imageData

    # JSON - "imageHeight" & "imageWidth"
    binary_mask = np.asarray(mask).astype(
        np.uint8
    )  # Image -> array ('uint-8') # binary_mask just contains 0 & 1
    JSON_OUTPUT["imageHeight"] = binary_mask.shape[0]  # height
    JSON_OUTPUT["imageWidth"] = binary_mask.shape[1]  # width

    # JSON - "shapes": label, points, group_id,  shape_type, flags
    success = False
    for i in np.unique(binary_mask):  # loop through each img's mask(frame)
        # Filter the masks(contains 'pneumonia')
        if i != 0:  # 0 : typically background
            # original mask_patch
            mask_tmp = np.where(
                binary_mask == i, 1, 0
            )  # check the connected area (set: 1, rest: 0 -> binarization)
            """
            Description for "np.where(condition, x(True), y(False))":
                mask_tmp = binary_mask
                for j in mask_tmp:
                    if j == i:
                        True
                        j = 1
                    else:
                        False
                        j = 0
                return mask_tmp: a new binary mask (contain only 0 & 1)
            """
            # Draw mask -> plot its boundary points
            # plt.subplot(1, 2, 1) # position: left
            # plt.imshow(mask_tmp, cmap='gray')

            # padded & filled mask_tmp's connected area
            mask_tmp = binary_dilation(
                mask_tmp, structure=np.ones((3, 3)), iterations=1
            )  # padding
            mask_tmp = binary_fill_holes(mask_tmp, structure=np.ones((5, 5)))  # filling
            # plt.subplot(1, 2, 2) # position: right
            # plt.imshow(mask_tmp, cmap='gray')

            # plt.show()

            # polygon extraction
            polygons = binary_mask_to_polygon(mask_tmp, tolerance=2)

            for polygon in polygons:
                if len(polygon) > 10:  # discard the contour if it contains few points
                    # "shapes" -> labels
                    label = class_names[i]  # class_names[1] : 'pneumonia'

                    # "shapes" -> points
                    # YOLO TXT -> one row, one contour(polygon)
                    points = []
                    yolotxt_row = str(class_mapping[label])  # YOLO txt_row -> classID
                    for j in range(0, len(polygon), 2):  # step_range: 2
                        x = polygon[j]  # default decimals = 1
                        y = polygon[j + 1]
                        points.append([x, y])  # point[x, y]

                        # YOLO txt_row -> point[x, y] -> normalization
                        x = round(x / JSON_OUTPUT["imageWidth"], 6)  # normalized
                        y = round(
                            y / JSON_OUTPUT["imageHeight"], 6
                        )  # normalized, 6: decimals = 6
                        yolotxt_row += " " + str(x) + " " + str(y)

                    # JSON - "shapes"
                    shape = {
                        "label": label,
                        "points": points,
                        "group_id": None,
                        # 'description': "",
                        "shape_type": "polygon",
                        "flags": {},
                        # 'mask' : None
                    }
                    JSON_OUTPUT["shapes"].append(shape)

                    # TXT - "row <-> contour"
                    TXT_OUTPUT.append(yolotxt_row)
                    # width_normalized, height_normalized ???

                    success = True

    return success, JSON_OUTPUT, TXT_OUTPUT


# Test Code

data_path = "./data/mosmed"
img_nii_path = os.path.join(data_path, "dataNii")
mask_nii_path = os.path.join(data_path, "maskNii")

# create data_preprocessing folder & sub_dir
data_pre_path = os.path.join(data_path, "data_preprocessing")
img_png_path = os.path.join(data_pre_path, "images")
mask_png_path = os.path.join(data_pre_path, "masks")
json_path = os.path.join(data_pre_path, "labelsJSON")
txt_path = os.path.join(data_pre_path, "labels")

PATH = {
    data_pre_path,  # preprocessing_tmp1 --> data augmentation for mask
    img_png_path,
    mask_png_path,
    json_path,
    txt_path
}
for path in PATH:
    os.makedirs(path, exist_ok=True)

class_names = ["_background_", "pneumonia"]
class_mapping = {"pneumonia": 0}

print("Starting: Nii -> PNG/JSON/TXT ...")

winwidth, wincenter = get_winwidth_wincenter(img_nii_path, mask_nii_path)

for nii_name in os.listdir(mask_nii_path):
    if nii_name.endswith('.nii.gz') or nii_name.endswith('.nii'):

        # read nii file
        imgs_nii = sitk.ReadImage(os.path.join(img_nii_path, nii_name))
        masks_nii = sitk.ReadImage(os.path.join(mask_nii_path, nii_name))

        # imgs_init = sitk.GetArrayFromImage(imgs_nii)

        # windowing
        imgs_nii = window_transform(imgs_nii, winwidth=winwidth, wincenter=wincenter)

        # nii -> array[]
        imgs = sitk.GetArrayFromImage(imgs_nii)
        masks = sitk.GetArrayFromImage(masks_nii).astype(np.uint8) # uint8 ???

        """
        # write nii's metadata in txt && Draw
        img_meta_path, mask_meta_path = create_metadata_dir(data_path)
        write_metadata_in_txt(f"{os.path.join(img_nii_path, nii_name)}", imgs_nii, imgs_init, img_meta_path)
        write_metadata_in_txt(f"{os.path.join(mask_nii_path, nii_name)}", masks_nii, masks, mask_meta_path)
        draw_image(imgs, 1, len(imgs), make_fig_name("image", nii_name))
        draw_histogram(imgs_init, 17, make_fig_name("image_init", nii_name, 17))
        """

        for idx in range(masks.shape[0]): # masks.shape: (depth, height, width)
            # get img/mask whose mask value > 0 -> which has pneumonia in CT
            if np.sum(masks[idx, :, :]) > 0: 
                img_png = Image.fromarray(imgs[idx, :, :]).convert('L') # array -> PIL Image object # L: RGB -> gray
                mask_png = Image.fromarray(masks[idx, :, :]).convert('L') # L: RGB -> gray

                # mask -> json
                sub_name = str(nii_name.split('.')[0] + '_' + str(idx))
                img_mask_json_path = os.path.join(json_path, sub_name + '.json' )
                img_mask_txt_path = os.path.join(txt_path, sub_name + '.txt' )
                success, JSON_OUTPUT, TXT_OUTPUT = img_mask_to_json_txt(img_png, mask_png, class_names, class_mapping)

                if success:
                    # 1.1 & 2.1 save PNG
                    img_png.save(os.path.join(img_png_path, sub_name + '.png'))
                    mask_png.save(os.path.join(mask_png_path, sub_name + '.png'))  # Save the original mask, not the mask after padding & filling & close_contours

                    # 2.2 write in JSON
                    # JSON - "imagePath"
                    JSON_OUTPUT['imagePath'] = sub_name + '.png'
                    with open(img_mask_json_path, 'w') as json_output:
                        json.dump(JSON_OUTPUT, json_output, indent=4) # Serialize python_data as a JSON file

                    # 2.3 write in TXT
                    with open(img_mask_txt_path, 'w') as txt_output:
                        TXT_OUTPUT = np.array(TXT_OUTPUT)
                        for idx, row in enumerate(TXT_OUTPUT):
                            if idx != len(TXT_OUTPUT) - 1:
                                row = row + '\n'
                            txt_output.write(row)

print("The conversion was completed successfully.")
