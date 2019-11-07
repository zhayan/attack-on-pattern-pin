import os

import sys
import cv2
import random
import skimage.io
from mrcnn.config import Config
from datetime import datetime
import videoToPic
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
import visualize
import numpy as np

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_9.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "test_data/input_images/")


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# 删除冗余结果
def delete_redu(r):
    max_phone_score = 0
    max_finger_score = 0
    n_instance = r['rois'].shape[0]
    instance = 0
    while instance < n_instance:
        if r['class_ids'][instance] == 1:
            if r['scores'][instance] > max_phone_score:
                max_phone_score = r['scores'][instance]
                generate_newbox(r, instance)
            else:
                r['rois'] = np.delete(r['rois'], instance, axis=0)
                r['class_ids'] = np.delete(r['class_ids'], instance, axis=0)
                r['scores'] = np.delete(r['scores'], instance, axis=0)
                r['masks'] = np.delete(r['masks'], instance, axis=2)
                instance -= 1
                n_instance -= 1
            instance += 1
        elif r['class_ids'][instance] == 2:
            if r['scores'][instance] > max_finger_score:
                max_finger_score = r['scores'][instance]
                # 生成新的box
                generate_newbox(r, instance)
            else:
                r['rois'] = np.delete(r['rois'], instance, axis=0)
                r['class_ids'] = np.delete(r['class_ids'], instance, axis=0)
                r['scores'] = np.delete(r['scores'], instance, axis=0)
                r['masks'] = np.delete(r['masks'], instance, axis=2)
                instance -= 1
                n_instance -= 1
            instance += 1


# calculate finger centre and generate box
def generate_newbox(r, instance):
    # 计算中心点
    print('开始计算中心点')
    mean_x = 0
    mean_y = 0
    total_x = 0
    total_y = 0
    point_count = 0
    for x in range(r['masks'].shape[0]):
        for y in range(r['masks'].shape[1]):
            if r['masks'][x, y, instance]:
                total_x += x
                total_y += y
                point_count += 1
    if point_count != 0:
        mean_x = total_x // point_count
        mean_y = total_y // point_count
    if r['masks'][mean_x, mean_y, instance]:
        print("开始计算新box")
        box = [mean_x, mean_y, mean_x, mean_y]
        while True:
            box[0] -= 1
            box[1] -= 1
            box[2] += 1
            box[3] += 1
            if r['masks'][box[0], box[1], instance] == False or r['masks'][
                box[2], box[3], instance] == False:
                break
        r['rois'][instance] = box


def export_txt(r, videoname, filename):
    with open(outputpath, "a+") as f:
        f.write(videoname + " " + filename[0:-4] + " ")
        
    i = 0
    for class_id in r['class_ids']:
        leftcornerx = r['rois'][i][0]
        leftcornery = r['rois'][i][1]
        width = r['rois'][i][2] - r['rois'][i][0]
        heigh = r['rois'][i][3] - r['rois'][i][1]
        with open(outputpath, "a+") as f:
            f.write(str(class_id) + " " + str(leftcornerx) + " " + str(leftcornery) + " " + str(width) + " " + str(heigh) + " ")
        
        i = i + 1
    with open(outputpath, "a+") as f:
        f.write("\n")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'phone', 'finger']
outputpath = "identification.txt"


if __name__ == '__main__':

    # caculate accuracy
    succ_count = 0
    rate = 0
    flag = 0
    # box to convert
    box = [0, 0, 0, 0]
    videoname = input("请输入要处理的名称:\n")
    videoToPic.main(videoname, 1)
    file_names = next(os.walk(IMAGE_DIR + videoname[0:11]))[2]
    for file in file_names:
        print('正在处理 ', file)

        # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
        a = datetime.now()

        # Run detection
        results = model.detect([image], verbose=1)
        b = datetime.now()

        # Visualize results
        print("time use", (b - a))
        r = results[0]

        # 删除冗余结果
        delete_redu(r)

        # 打印结果个数
        rois = np.array(r['rois'])
        print(rois.shape)

        # 若成功则输出到文件
        if rois.shape[0] == 2:
            print("成功找到手指")
            flag = 1
            # 输出到txt
            export_txt(r, videoname, file)
            # 显示图像
            image = visualize.display_instances(file, image, r['rois'], r['masks'], r['class_ids'],
                                            class_names, r['scores'])
            break

    # 若失败则输出fail
    if not flag:
        with open(outputpath, "w") as f:
            f.write("fail")

    # 计算成功率
    # rate = round(succ_count / all_count, 3)
    # print("success rate:  ", rate)
