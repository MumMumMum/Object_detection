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


"""Convert raw PASCAL dataset to TFRecord for object_detection.
Example usage:
    ./create_pascal_tf_record --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""


import tensorflow as tf
import yaml
import os
import dataset_util
import io

from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('input_path', 'additional_train.yaml', 'Path to the yaml input')
flags.DEFINE_string('output_path', 'train_TFRecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS


LABEL_DICT = {
 "GreenRight": 1,
        "RedLeft":2,
        "RedStraight": 3,
        "GreenStraightLeft":4,
        "occluded":5,
        "GreenStraightRight": 6,
        "RedRight":7,
        "Green": 8,
        "Yellow": 9,
        "Red":10,
        "GreenLeft": 11,
        "GreenStraight": 12,
        "RedStraightLeft": 13,
        "off":14

}

def create_tf_example(example):
    filename = example['path'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()
    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    width, height = image.size
    print("w",width,"h",height)
        
    image_format = 'png'.encode() 
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    for box in example['boxes']:
        #if box['occluded'] is False:
        #print("adding box")
        xmins.append(float(box['x_min'] / width))
        xmaxs.append(float(box['x_max'] / width))
        ymins.append(float(box['y_min'] / height))
        ymaxs.append(float(box['y_max'] / height))
        classes_text.append(box['label'].encode())
        classes.append(int(LABEL_DICT[box['label']]))
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example
    
    
 
def main(_):
    print(os.path.join(os.getcwd(), FLAGS.output_path))
    #print((os.getcwd()))
    #print(FLAGS.input_path)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples = yaml.load(open(os.path.join(os.getcwd(), FLAGS.input_path), 'rb').read())
    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")
    
    #examples = examples[:10]  # for testing
    # is dict with key as box and path
    #print("example" , examples[8])
    for i in range(len(examples)):
        examples[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(FLAGS.input_path), examples[i]['path']))
        #print("example" , examples[i]['path'])
        
        
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()
    
    
if __name__ == '__main__':
    tf.app.run()
    
    
    