# improved_yolov5_for_multi_scale_traffic_sign_detection

This repository provides a comprehensive guide to setting up and using YOLOv5 for traffic sign detection. The following steps will guide you through the installation of necessary Python packages, setting up Google Drive for data storage, cloning the YOLOv5 module, and preparing your dataset.

## Some Modifications in the Project Codebase.
Update Version of the Python Packages.
Difference Ratio of the Training Datasets & Test DataSets(105: 100)
Datasets Images Height & Weight Measurement
Label of the data that was trained on the datasets. (Traffic Sign Data).
We trained the model large Datasets Image.
Show Tensorboard Where Precision, Recall, Confusion Matrix, and other graphs
will be shown based on the datasets.
Detect Traffic Signs on the Video Datasets.
Show Plot Diagrams and Charts Based on the datasets.


## Table of Contents

1. [Install All Modules Of Python Packages](#install-all-modules-of-python-packages)
2. [Connect Google Drive](#connect-google-drive)
3. [Clone YOLOv5 Module](#clone-yolov5-module)
4. [Download and Prepare the Dataset](#download-and-prepare-the-dataset)
5. [Train the Model](#train-the-model)
6. [Inference and Results](#inference-and-results)
7. [Visualizing Results](#visualizing-results)

## Install All Modules Of Python Packages

To begin, ensure that all required Python packages are installed. This includes specific versions of TensorFlow, TensorBoard, and protobuf, among others. It is recommended to upgrade pip before installing these packages to avoid any conflicts.

```sh
!python -m pip install --upgrade pip
!pip install tensorflow==2.16.1
!pip install tensorboard==2.16.2
!pip install protobuf==4.25.3
```

## Connect Google Drive

For this project, we will use Google Drive to store our dataset and models. Mount your Google Drive to the Colab environment.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Change the directory to where you have your image detection files in Google Drive.

```python
%cd '/content/drive/My Drive/image_detection'
```

## Clone YOLOv5 Module

Next, clone the YOLOv5 repository from GitHub. This repository contains the YOLOv5 implementation which we will use for training and inference.

```sh
!git clone https://github.com/ultralytics/yolov5
```

## Download and Prepare the Dataset

Download the dataset for traffic sign detection. Ensure that the dataset is organized in the correct format required by YOLOv5.

```sh
# Example: Download a dataset from a URL or Google Drive
!wget -P /content/drive/My\ Drive/image_detection/dataset/ <dataset_url>
```

Unzip and organize the dataset into `train`, `val`, and `test` directories.

```sh
!unzip /content/drive/My\ Drive/image_detection/dataset/<dataset.zip> -d /content/drive/My\ Drive/image_detection/dataset/
```

## Train the Model

With the dataset prepared, you can now train the YOLOv5 model. Ensure you are in the YOLOv5 directory and execute the training script with the necessary configurations.

```sh
%cd yolov5
!python train.py --img 640 --batch 16 --epochs 50 --data /path/to/dataset.yaml --weights yolov5s.pt
```

## Inference and Results

After training, you can use the model for inference on test images. Use the detect script provided in the YOLOv5 repository.

```sh
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source /path/to/test/images
```

## Visualizing Results

Finally, visualize the detection results to see how well your model performs on the test images.

```python
from IPython.display import Image, display
display(Image(filename='/path/to/result_image.jpg'))
```

## Conclusion

This guide provides a step-by-step approach to setting up YOLOv5 for traffic sign detection. Follow the steps carefully, and refer to the YOLOv5 documentation for further customization and advanced usage.

For any issues or contributions, feel free to open an issue or pull request on the GitHub repository. Happy coding!# Project-Code-YoloV5
