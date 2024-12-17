"""
data:
    CT         :     used
    mask       :     used
    labels(txt): not used
    labelsJson : not used
"""


# U-Net Structure:
# 1->64->64...............................................................->(*/)128->64->64=>2      # 1: input image, 2: output segmentation map
#    (-+)64->128->128...........................................->(*/)256->128->128
#             (-+)128->256->256.......................->(*/)512->256->256
#                       (-+)256->512->512..->(*/)1024->512->512     # 1024: 512+512
#                                 (-+)512->1024->1024
# ->  : conv 3x3, RELU
# ..->: copy & crop
# (-+): max pool 2x2
# (*/): up-conv 2x2
# =>  : conv 1x1

'''
Issues:
1. 要不直接把preprocessing_tmp1 经过"data_generation"分割90% 出来用作训练集, 并存为dataset放在外面目录下
 ----> 看下之后test部分的图片怎么预处理, 如果处理方式一样那就dataset拿出来放到外面去
'''

from keras._tf_keras import keras  # CPU - keras > 3.*
from keras._tf_keras.keras.layers import *  # CPU - keras > 3.*
from keras._tf_keras.keras.preprocessing.image import (
    ImageDataGenerator,
)  # CPU - keras > 3.*

# from keras.layers import *  # GPU - keras > 2.*
# from keras.callbacks import ModelCheckpoint  # GPU - keras > 2.*
# from keras.preprocessing.image import ImageDataGenerator  # GPU - keras > 2.*

from keras import Model
from keras import backend as K

import os
import numpy as np
# from data_preparation import draw_image
import matplotlib.pyplot as plt
import cv2


# img & mask
# 3. data augmentation: (import tensorflow.keras.preprocessing.Image)
#         [1] define an image_generator -> ImageDataGenerator()
#         [2] image data augmentation -> flow_from_directory()
#         [3] image normalization
#     问题:
#         [1] 先进行.nii -> png/json/txt, 后进一步keras数据增强
#             有个问题: images/mask增强后随之的json/txt是否也要发生改变 ---> ???
#         [2] tensorflow和torch一起用 ---> 可以
#             model -> Yolo 使用的是pytorch
#             data augmentation 使用的是 tensorflow->keras
#         [3] gene后一定要跟fit()均值化，否则会提示：
#               F:\AI_Outils\Anaconda\1\envs\opencv_CPU\Lib\site-packages\keras\src\legacy\preprocessing\image.py:1263: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.


# data augmentation for train
def train_generator(dataset_path, type):
    data_path = os.path.join(dataset_path, type)
    data_pre_path = os.path.join(dataset_path, f"{type}_generator")
    img_png_path = os.path.join(data_pre_path, "images")
    mask_png_path = os.path.join(data_pre_path, "masks")

    PATH = {
        data_pre_path,
        img_png_path,
        mask_png_path,
    }
    for path in PATH:
        os.makedirs(path, exist_ok=True)

    # 3.1 define an image_generator: to perform various transformations on object
    generator_args = dict(
        rotation_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=False,
        vertical_flip=False,
    )
    generator_image = ImageDataGenerator(generator_args)
    generator_mask = ImageDataGenerator(generator_args)

    # 3.2 implement further data augmentation for image & mask
    generation_image = generator_image.flow_from_directory(
        directory=data_path,
        classes=["images"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(512, 512),
        batch_size=2,
        save_to_dir=os.path.join(data_pre_path, "images"),
        # save_prefix='ct_',
        seed=123,
    )
    generation_mask = generator_mask.flow_from_directory(
        directory=data_path,
        classes=["masks"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(512, 512),
        batch_size=2,
        save_to_dir=os.path.join(data_pre_path, "masks"),
        # save_prefix='mask_',
        seed=123,
    )
    generation = zip(generation_image, generation_mask)

    print("2--------------------------")
    i = 0
    # 3.3 image normalization (image -> not normalized yet, mask -> binary)
    for image, mask in generation:
        '''
        i = i + 1

        # output image data to TXT
        arr = np.array(image[0][:, :, 0])
        np.savetxt("array_0.txt", arr)
        print(f"image: min: {np.nanmin(arr)}, max: {np.nanmax(arr)}.")

        # output image data to TXT
        arr = np.array(normalization(image)[0][:, :, 0])
        np.savetxt("array_1.txt", arr)
        print(
            f"normalization_image: min: {np.nanmin(arr)}, max: {np.nanmax(arr)}."
        )

        print(image.shape, mask.shape)  # (2, 256, 256, 1) (2, 256, 256, 1) --> batch_size=2

        # visualization
        data_img_slices = [
            image[0][:, :, 0],
            normalization(image)[0][:, :, 0],
            mask[0][:, :, 0],
            image[1][:, :, 0],
            normalization(image)[1][:, :, 0],
            mask[1][:, :, 0],
        ]
        draw_image(data_img_slices, 1, 6, None)
        
        if i == 1:
            break
        '''

        yield (normalization(image), mask)
        # image[0][:, :, 0] = normalization(image[0][:, :, 0])
        # image[1][:, :, 0] = normalization(image[1][:, :, 0])
    print("Further data augmentation was completed successfully.")


def binarization(data):
    """
    Binarization: Converts data to only two values, e.g. 0 & 1
    To do: To highlight certain features in the image
    Processing: x'[x/255.0 > 0.5] = 1.0
                x'[x/255.0 <= 0.5] = 0.0
    """
    data_binary = data / 255.0
    data_binary[data_binary > 0.5] = 1.0
    data_binary[data_binary <= 0.5] = 0.0
    return data_binary


def standardization(data):
    """
    Standardization: Converts the data into a new distribution with a mean of 0 and a standard deviation of 1
    To do: to have comparability between different features (if data feature value range/unit is quite different --> perform standardization).
        --> Standardization does not change the distribution of feature data
    Processing: x' = (x - mean) / std
    """
    mean = np.mean(data)
    std = np.std(data)
    data_std = (data - mean) / std
    return data_std


def normalization(data):
    """
    Normalization: Scale the data to a specific range, e.g. 0-1
    To do: To make the influence of each feature on the target variable consistent.
        --> Data normalization changes the distribution of feature data
    Processing: x' = (x - min)/(max - min)
    Note: In the medical field, normalization is generally performed --> to accelerate the convergence of the network and make the model more stable
    """
    min = np.nanmin(data)
    max = np.nanmax(data)
    data_nor = (data - min) / (max - min)
    return data_nor


def u_net(input_size = (512, 512, 1), path=None):
    # layer 1-1
    inputs_L1_1 = Input(input_size)
    conv1_L1_1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs_L1_1)  # filters: 64, kernel_size:3x3, kernel_initializer: use normal distribution to initializer Weights of kernel
    conv2_L1_1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1_L1_1)
    pool1_L1_1 = MaxPool2D(pool_size=(2,2))(conv2_L1_1)

    # layer 2-1
    conv3_L2_1 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1_L1_1)
    conv4_L2_1 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3_L2_1)
    pool2_L2_1 = MaxPool2D(pool_size=(2, 2))(conv4_L2_1)

    # layer 3-1
    conv5_L3_1 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2_L2_1)
    conv6_L3_1 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5_L3_1)
    pool3_L3_1 = MaxPool2D(pool_size=(2, 2))(conv6_L3_1)

    # layer 4-1
    conv7_L4_1 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3_L3_1)
    conv8_L4_1 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7_L4_1)
    pool4_L4_1 = MaxPool2D(pool_size=(2, 2))(conv8_L4_1)

    # layer 5
    conv9_L5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4_L4_1)
    conv10_L5 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9_L5)
    up1_L5 = UpSampling2D(size=(2, 2))(conv10_L5)  # deconvolution

    # layer 4-2
    conv11_L4_2 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(concatenate([up1_L5, conv8_L4_1], axis=3))  # concatenation
    conv12_L4_2 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv11_L4_2)
    up2_L4_2 = UpSampling2D(size=(2, 2))(conv12_L4_2)

    # layer 3-2
    conv13_L3_2 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(concatenate([up2_L4_2, conv6_L3_1], axis=3))
    conv14_L3_2 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv13_L3_2)
    up3_L3_2 = UpSampling2D(size=(2, 2))(conv14_L3_2)

    # layer 2-2
    conv15_L2_2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(concatenate([up3_L3_2, conv4_L2_1], axis=3))
    conv16_L2_2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv15_L2_2)
    up4_L2_2 = UpSampling2D(size=(2, 2))(conv16_L2_2)

    # layer 1-2
    conv17_L1_2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(concatenate([up4_L2_2, conv2_L1_1], axis=3))
    conv18_L1_2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv17_L1_2)
    outputs_L1_2 = Conv2D(1, 1, activation="sigmoid")(conv18_L1_2)

    # build model
    model = Model(inputs = inputs_L1_1, outputs = outputs_L1_2)

    # compile model
    '''
    Loss     : 0-1 binary cross-entropy (binary_crossentropy)
    Optimizer: Adaptive Descent (Adam)
    Callback : After each epoch is trained, autosave a best pre-trained model(optimal weights). (keras.callbacks.ModelCheckpoint)
    '''
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


class ShowMask(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print()
        idx = 0
        for img, mask in gene:
            compare_list = [img[0], mask[0], model.predict(img[0].reshape(1, 512, 512, 1))[0]]
            for i in range(0, len(compare_list)):
                plt.subplot(1, 3, i+1)
                plt.imshow(compare_list[i], cmap="gray")
                plt.axis(False)
            # plt.show()
            plt.savefig(f"compare_{idx}.png")
            idx = idx + 1
            break
        # return super().on_epoch_end(epoch, logs)


# Test

# dataset: data augmentation for train data
UNETDataset_path = "./UNETDataset"
gene = train_generator(UNETDataset_path, "train")  # could be used as input to the model and directly as training

# model params
steps_per_epoch = 50
epochs = 100
model_name = f"u_net-512-512-1-pneumonia_{epochs}_{steps_per_epoch}.keras"
models_path = "./models/"
model_path = os.path.join(models_path, model_name)
model_ckpt = keras.callbacks.ModelCheckpoint(model_path, save_best_only=False, verbose=1)

# train
K.clear_session() # keras

model = u_net(path=model_path) # structure
# print(model.summary())

# model.fit(gene, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_ckpt, ShowMask()]) # train
model.fit(
    gene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_ckpt],
)  # train



# 思路：
# 1. Dataset_mini -> CPU OK, GPU KO
# 2. cudnn v8 -> GPU KO
# 3. 缩减 input_size -> 512->256
# 4. 缩减 unet structure

# evalution
data_val_generator_path = os.path.join(UNETDataset_path, "val_generator")
compare_path = os.path.join(data_val_generator_path, "compare")
PATH = {data_val_generator_path, compare_path}
for path in PATH:
    os.makedirs(path, exist_ok=True)

# model_test= keras.models.load_model(model_path)
'''
gene = train_generator(UNETDataset_path, "val")
idx = 0
for img, mask in gene:
    predict_mask = model_test.predict(img)[0]
    # predict_mask_np = (predict_mask * 255).numpy()

    _, real_mask = cv2.threshold(mask[0], 127, 255, 0)
    real_mask = (real_mask).astype('uint8')
    real_contours, _ = cv2.findContours(real_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    real_overlap_img = cv2.drawContours(img[0].copy(), real_contours, -1, (0, 255, 0), 2)

    _, pred_mask = cv2.threshold((predict_mask * 255).astype("uint8"), 127, 255, 0)
    pred_mask = (pred_mask).astype('uint8')
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_overlap_img = cv2.drawContours(img[0].copy(), pred_contours, -1, (255, 0, 0), 2)

    compare_list = [img[0], pred_mask, real_overlap_img, pred_overlap_img]

    for i in range(0, len(compare_list)):
        plt.subplot(1, 4, i+1)
        plt.imshow(compare_list[i], cmap="gray")
        plt.axis(False)
    # plt.show()
    save_path = os.path.join(compare_path, f"compare_{idx}.png")
    plt.savefig(save_path)

    idx = idx + 1
'''

'''
# test save compare_png

images_path = os.path.join(data_val_generator_path, "images")
masks_path = os.path.join(data_val_generator_path, "masks")

idx = 0
for png_name in os.listdir(images_path):
    # predict_mask = model_test.predict(img)[0]
    
    img = cv2.imread(os.path.join(images_path, png_name))
    mask = cv2.imread(os.path.join(masks_path, png_name), cv2.IMREAD_GRAYSCALE)

    _, real_mask = cv2.threshold(mask, 127, 255, 0)
    real_mask = (real_mask).astype("uint8")
    real_contours, _ = cv2.findContours(
        real_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    real_overlap_img = cv2.drawContours(
        img.copy(), real_contours, -1, (0, 255, 0), 2
    )

    compare_list = [img, real_overlap_img]

    for i in range(0, len(compare_list)):
        plt.subplot(1, 2, i + 1)
        plt.imshow(compare_list[i], cmap="gray")
        plt.axis(False)
    # plt.show()
    save_path = os.path.join(compare_path, f"compare_{idx}.png")
    plt.savefig(save_path)

    idx = idx + 1
'''