""" ++++++++++++++++++++++ USE THIS CODE TO TRAIN MODEL FROM SCRATCH (WITHOUT USING COCO WEIGHTS) ++++++++++++++++++++++ """

# Import Libraries

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import imgaug
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project

ROOT_DIR = "X:/ML1/WEEK8/day4/task4_2/Mask-R-CNN-using-Tensorflow2"

# Import Mask R-CNN
# Append the ROOT_DIR to the system path to find the local version of the Mask R-CNN library

sys.path.append(ROOT_DIR)

# Directory to save logs and model checkpoints during training

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Define the configuration for the custom Mask R-CNN model

class CustomConfig(Config):
    """Configuration for training on the custom dataset."""
    NAME = "object"  # Name of the configuration
    GPU_COUNT = 1  # Number of GPUs to use (1 if you're using a single GPU)
    IMAGES_PER_GPU = 1  # Number of images to process per GPU (adjust based on your GPU memory)
    NUM_CLASSES = 1 + 2  # Background + 2 classes (cat and dog)
    STEPS_PER_EPOCH = 5  # Number of steps per epoch (adjust based on your dataset size)
    DETECTION_MIN_CONFIDENCE = 0.9  # Minimum confidence threshold for detections
    LEARNING_RATE = 0.001  # Learning rate for training

# Define the dataset class for loading and processing the custom dataset

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset (either 'train' or 'val')."""
        # Add classes (background is added by default as class 0)
        self.add_class("object", 1, "cat")
        self.add_class("object", 2, "dog")

        # Ensure the subset is either 'train' or 'val'
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load the annotations file (in JSON format)
        annotations_file = os.path.join(dataset_dir, 'annotations.json')
        with open(annotations_file) as f:
            annotations_data = json.load(f)

        # Filter out annotations that have 'regions' defined
        annotations = [a for a in annotations_data.values() if 'regions' in a and a['regions']]

        # Iterate over each annotation
        for a in annotations:
            # Extract polygons and object labels from the annotation
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [r['region_attributes']['label'] for r in a['regions'].values()]
            name_dict = {"cat": 1, "dog": 2}  # Map object names to class IDs
            num_ids = [name_dict[obj] for obj in objects]

            # Load the corresponding image and extract its dimensions
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # Add the image to the dataset
            self.add_image(
                "object",  # Source name
                image_id=a['filename'],  # Image ID
                path=image_path,  # Image path
                width=width, height=height,  # Image dimensions
                polygons=polygons,  # Polygons for masks
                num_ids=num_ids  # Class IDs for each object in the image
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Create a mask array with zeros
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        # Iterate over polygons to draw masks
        for i, p in enumerate(info["polygons"]):
            # Get the pixels inside the polygon
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            # Ensure the indices are within the bounds of the mask
            rr = np.clip(rr, 0, info["height"] - 1)
            cc = np.clip(cc, 0, info["width"] - 1)

            # Set the mask for the current polygon
            mask[rr, cc, i] = 1

        # Convert list of class IDs to a numpy array
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# Define the training function

def train(model):
    """Train the Mask R-CNN model on the custom dataset."""
    # Load the training dataset
    dataset_train = CustomDataset()
    dataset_train.load_custom("dataset", "train")
    dataset_train.prepare()

    # Load the validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("dataset", "val")
    dataset_val.prepare()

    # Start training the model
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,  # Set learning rate
                epochs=100,  # Number of epochs to train
                layers='heads')  # Specify which layers to train

# Initialize the configuration and model

config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# No weights to load, training from scratch

train(model)















""" ++++++++++++++++++++++ USE THIS CODE TO TRAIN MODEL USING COCO WEIGHTS ++++++++++++++++++++++ """

# Import Libraries

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import imgaug
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project

ROOT_DIR = "X:/ML1/WEEK8/day4/task4_2/Mask-R-CNN-using-Tensorflow2"

# Import Mask R-CNN
# Append the ROOT_DIR to the system path to find the local version of the Mask R-CNN library

sys.path.append(ROOT_DIR)

# Path to trained weights file

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints during training

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Define the configuration for the custom Mask R-CNN model

class CustomConfig(Config):
    """Configuration for training on the custom dataset."""
    NAME = "object"  # Name of the configuration
    GPU_COUNT = 1  # Number of GPUs to use (1 if you're using a single GPU)
    IMAGES_PER_GPU = 1  # Number of images to process per GPU (adjust based on your GPU memory)
    NUM_CLASSES = 1 + 2  # Background + 2 classes (cat and dog)
    STEPS_PER_EPOCH = 5  # Number of steps per epoch (adjust based on your dataset size)
    DETECTION_MIN_CONFIDENCE = 0.9  # Minimum confidence threshold for detections
    LEARNING_RATE = 0.001  # Learning rate for training

# Define the dataset class for loading and processing the custom dataset

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset (either 'train' or 'val')."""
        # Add classes (background is added by default as class 0)
        self.add_class("object", 1, "cat")
        self.add_class("object", 2, "dog")

        # Ensure the subset is either 'train' or 'val'
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load the annotations file (in JSON format)
        annotations_file = os.path.join(dataset_dir, 'annotations.json')
        with open(annotations_file) as f:
            annotations_data = json.load(f)

        # Filter out annotations that have 'regions' defined
        annotations = [a for a in annotations_data.values() if 'regions' in a and a['regions']]

        # Iterate over each annotation
        for a in annotations:
            # Extract polygons and object labels from the annotation
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [r['region_attributes']['label'] for r in a['regions'].values()]
            name_dict = {"cat": 1, "dog": 2}  # Map object names to class IDs
            num_ids = [name_dict[obj] for obj in objects]

            # Load the corresponding image and extract its dimensions
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # Add the image to the dataset
            self.add_image(
                "object",  # Source name
                image_id=a['filename'],  # Image ID
                path=image_path,  # Image path
                width=width, height=height,  # Image dimensions
                polygons=polygons,  # Polygons for masks
                num_ids=num_ids  # Class IDs for each object in the image
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Create a mask array with zeros
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        # Iterate over polygons to draw masks
        for i, p in enumerate(info["polygons"]):
            # Get the pixels inside the polygon
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            # Ensure the indices are within the bounds of the mask
            rr = np.clip(rr, 0, info["height"] - 1)
            cc = np.clip(cc, 0, info["width"] - 1)

            # Set the mask for the current polygon
            mask[rr, cc, i] = 1

        # Convert list of class IDs to a numpy array
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# Define the training function

def train(model):
    """Train the Mask R-CNN model on the custom dataset."""
    # Load the training dataset
    dataset_train = CustomDataset()
    dataset_train.load_custom("dataset", "train")
    dataset_train.prepare()

    # Load the validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("dataset", "val")
    dataset_val.prepare()

    # Start training the model
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,  # Set learning rate
                epochs=100,  # Number of epochs to train
                layers='heads')  # Specify which layers to train

# Initialize the configuration and model

config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

train(model)