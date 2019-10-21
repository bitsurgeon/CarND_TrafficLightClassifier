# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert Udacity Capstone site dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_capstone_tf_record.py \
        --data_dir=/home/user/capstone_dataset \
        --output_path=/home/user/capstone.record
"""

import os
import io
import hashlib
import yaml
import logging
import PIL.Image
import tensorflow as tf
import matplotlib.image as mpimg
from object_detection.utils import dataset_util, label_map_util
from lxml import etree

flags = tf.app.flags
flags.DEFINE_string('data_dir', '',
                    'Specify root directory to raw dataset. Separate multiple datasets with a comma.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory. (Needed for XML!)')
flags.DEFINE_string('output_path', '',
                    'Path to output TFRecord e.g.: data/train.record')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto e.g.: data/label_map.pbtxt')
flags.DEFINE_boolean('ignore_difficult_instances', False,
                     'Whether to ignore difficult instances')
FLAGS = flags.FLAGS


def create_tf_record(data, label_map_dict, is_yaml=False, ignore_difficult_instances=False):
    """
    Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
    :param data: dict holding (XML or YAML) fields for a single image (obtained by running dataset_util.recursive_parse_xml_to_dict)
    :param label_map_dict: A map from string label names to integers ids.
    :param ignore_difficult_instances: Whether to skip difficult instances in the dataset  (default: False).

    Returns:
    :return tf_example: The converted tf.Example.

    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    with tf.gfile.GFile(data['path'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    if is_yaml:
        width = int(data['width'])
        height = int(data['height'])
        filename = data['path'].encode('utf8')
        for box in data['boxes']:
            difficult_obj.append(0)

            xmin.append(float(box['x_min']) / width)
            ymin.append(float(box['y_min']) / height)
            xmax.append(float(box['x_max']) / width)
            ymax.append(float(box['y_max']) / height)
            classes_text.append(box['label'].encode('utf8'))
            classes.append(label_map_dict[box['label']])
            truncated.append(0)
            poses.append(r'Unspecified'.encode('utf8'))
    else:
        width = int(data['size']['width'])
        height = int(data['size']['height'])
        filename = data['filename'].encode('utf8')

        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(r'jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return tf_example

def main(_):

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    
    data_dir_list = FLAGS.data_dir.split(',')

    for dataset in data_dir_list:
        logging.info('Reading dataset in PASCAL format from %s', dataset)

        annotations_dir = os.path.join(dataset, FLAGS.annotations_dir)
        examples_list = [os.path.splitext(name)[0] for name in os.listdir(
            dataset) if os.path.isfile(os.path.join(dataset, name))]
        for example in examples_list:
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = create_tf_record(
                data, label_map_dict, ignore_difficult_instances=FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
