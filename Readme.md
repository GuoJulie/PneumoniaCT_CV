# 1. Goal: 使用 YoloV8 对医学影像CT/磁共振图像进行肺炎区域分割:

### part1 :
A. init data [nii(Nifit) -> png]: 将CT原始数据从 三维转二维, 保存png格式
B. mask data [nii -> json]: 将CT的分割标注数据从 三维转二维, 再转Labelme可以识别的json文件

### part2:
**Algorithm: Yolo-V8**
A. make dataset [train/val/test + label]: 对二维数据和json文件进行数据划分, 并json转换Yolo标签格式
B. Segmentation [Yolo-V8]: 使用YoloV8进行分割



# 2. JSON format (eg:) :
{
  "version": "4.5.6",  // Labelme software version
  "flags": {},         // Additional flag information
  "shapes": [
    {
      "label": "dog",  // Label name
      "points": [      // Polygon vertex points of the annotation 
        [298, 151],
        [324, 151],
        [324, 160],
        [317, 160],
        [317, 168],
        [307, 168],
        [307, 151],
        [298, 151]
      ],
      "group_id": null,
      "shape_type": "polygon",  // Shape type
      "flags": {}
    },
    {
      "label": "cat",
      "points": [
        [400, 200],
        [425, 200],
        [425, 210],
        [415, 210],
        [415, 220],
        [405, 220],
        [405, 200],
        [400, 200]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
    ...
  ],
  "imagePath": "path/to/image.jpg",  // Image file path
  "imageData": null,                 // Image data (Base64 encoded)
  "imageHeight": 480,                // Image height
  "imageWidth": 640                  // Image width
}





YOLO .txt format:
one object -> one row
(classID, x_center, y_center, width_normalized, height_normalized)

(eg):
1 0.375000 0.625000 0.312500 0.416667
0 0.656250 0.291667 0.156250 0.208333





dataset.yaml format(eg):
train: /path/to/your/dataset/images/train
val: /path/to/your/dataset/images/val
nc: 5
names: ['cat', 'dog', 'person', 'car', 'truck']