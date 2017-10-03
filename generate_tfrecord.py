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
    python generate_tfrecord.py G:\CarND\ObjectDetection\BoschData\train_01\train.yaml G:\CarND\ObjectDetection\BoschData\train_01\train_045.record
G:\CarND\ObjectDetection\BoschData\train_01\train_045.record train

python generate_tfrecord.py G:\CarND\ObjectDetection\BoschData\train_01\train.yaml G:\CarND\ObjectDetection\BoschData\test_01\test_045.record
G:\CarND\ObjectDetection\BoschData\test_01\test_045.record test
"""

## 3rd param train stands for train, test for test
import tensorflow as tf
import yaml
import os
import dataset_util
import io

import sys
import logging

from PIL import Image

flags = tf.app.flags

#flags.DEFINE_string('input_path_dir', '\\train_01', 'Path to the yaml input')
#flags.DEFINE_string('input_path_dir', '\\test_01', 'Path to the yaml input')
FLAGS = flags.FLAGS


LABEL_DICT = {
        "off":1,
		"Red":2,
		"Yellow": 3,
		"Green": 4,
		"GreenRight": 5,
        "RedLeft":6,
        "RedStraight": 7,
        "GreenStraightLeft":8,
        "occluded":9,
        "GreenStraightRight": 10,
        "RedRight":11,
        "GreenLeft": 12,
        "GreenStraight": 13,
        "RedStraightLeft": 14,
        

}

def create_tf_example(example):
    filename = example['path'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()
    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = Image.open(encoded_image_io)
    width, height = image.size
    #print("w",width,"h",height)
        
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
    
    writer = tf.python_io.TFRecordWriter(record_file)
    examples = yaml.load(open(yaml_file, 'rb').read())

    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")
    
    examples = examples[:10]  # for testing
    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")
    path = os.getcwd()+str_path
    print("path is ",path)
    for i in range(len(examples)):
        print("path is ",os.path.abspath(os.path.join(path, examples[i]['path'])))
        examples[i]['path'] = os.path.abspath(os.path.join(path, examples[i]['path']))
        #print(examples[i])
        
    counter = 0   
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
        if counter % 10 == 0:
            print("percent done",(counter/len(examples)*100))
        
        counter += 1
    writer.close()
    print("serialization done!!!")
    
    
if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(-1)
        
    yaml_file    = sys.argv[1]
    record_file =  sys.argv[2]
    train = sys.argv[3]
    print("trains",train)
    if train == 'train' :
        str_path = '\\train_01'
    else :
        str_path = '\\test_01'
    print (record_file)
    tf.app.run()
    
    
    