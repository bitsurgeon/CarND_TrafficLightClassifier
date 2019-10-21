# Traffic Light Classifier

## _Traffic light detector module of the capstone project of Self-Driving Car Engineer Nanodegree_

[![Udacity - Self-Driving Car Engineer Nanodegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  

This repository documents the procedures on training a SSD (Single Shot Detector) model for traffic light state classification using learning transfer from pre-trained MobileNet models using Google TensorFlow Object Detection API.

---

## Environment

- clone tensorflow object detection model [repo](https://github.com/tensorflow/models.git)
- install the model following the official [guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- this [troubleshooting](https://github.com/alex-lechner/Traffic-Light-Classification#troubleshooting) guide can be helpful
- assuming [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) has been installed

## Dataset

Images from simulator and test site need to be collected in this step.

- Simulator: save camera images in the callback function for `/image_color` ROS topic.
- Site: extract images from rosbag provided by Udacity.

To save images from rosbag, make sure `image_view` has been installed with ROS Kinetic packages, or it can be installed with the following command:  

```sh
sudo apt-get install ros-kinetic-image-view
```

Use the following commands to capture the images:

```sh
# 1st terminal
roscore

# 2nd terminal
rosbag play -l path/to/your_rosbag_file.bag

# 3rd terminal
rosrun image_view image_saver image:=/image_raw
```

Select images with traffic light in different distance and states for training. Make sure the numbers of different color states are balanced.

In addition to the images captured in above, publicly available datasets can be incorporated to derive better models.
- Bosch Small Traffic Lights [Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)
- Past Udacity Student - [Alex](https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0)
- Past Udacity Student - [Vatsal](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)

## Labeling

This is probably the most time consuming and boring step in the whole model training process. But please don't get fooled. How well the images get labeled in this step will have direct effect on the model accuracy at the end.

Among many image labeling [tools](https://hackernoon.com/the-best-image-annotation-platforms-for-computer-vision-an-honest-review-of-each-dac7f565fea), the one I used is [labelImg](https://github.com/tzutalin/labelImg). It's interface is pretty straightforward, and easily to get job done.

Annotations are saved in  `.xml` file. And it is in [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) format, which is the same format used by [ImageNet](http://www.image-net.org/). Each image will have its own annotation file in `labels` folder. Each light state can have its own folders.

Note that the labeled image from Bosch dataset is in `.yaml` file format.

## TFRecord

A [TFRecord](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) file is needed to retrain a TensorFlow model. It is a binary file that stores images and ground truth annotations. All necessary scripts are provided by TensorFlow.

The following items are required to generate a TFRecord file:

- a [`label_map.pbtxt`](./label_map.pbtxt) file that contains the labels (e.g. Red, Green or Yellow) with an associated ID (which must start from 1 instead of 0)
- the labeled `.xml` files
- script `create_tf_record.py` from TensorFlow [dataset tools](https://github.com/tensorflow/models/tree/master/research/object_detection/dataset_tools)

For this project, a new [script](./create_capstone_tf_record.py) is created to generate our TFRecord file.

```sh
python object_detection/dataset_tools/create_capstone_tf_record.py --data_dir=path/to/green/lights,path/to/red/lights,path/to/yellow/lights --annotations_dir=labels --output_path=your/path/to/train.record --label_map_path=path/to/your/label_map.pbtxt
```

The generated TFRecord file is expected to be similar size as the total size of the images.

If Bosch dataset is used, please refer to this [link](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d).

## Model