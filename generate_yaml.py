import tensorflow as tf
import yaml
import os

import io
from os import listdir
from os.path import isfile, join
from PIL import Image

import sys
import logging

"""python generate_yaml.py G:\CarND\ObjectDetection\BoschData\test_01\rgb\test G:\CarND\ObjectDetection\BoschData\test_01\test_01.yaml

"""
flags = tf.app.flags
#flags.DEFINE_string('input_path', '\\test_01', 'Path to the folder test ')
#flags.DEFINE_string('output_path', 'test_01.yaml', 'Path to output yaml ')
#flags.DEFINE_string('input_path1', 'additional_train.yaml', 'Path to output yaml ')
FLAGS = flags.FLAGS

def main(_):
    
    str= '\\rgb\\test\\'
  
    onlyfiles = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
    
   
    
    str= './rgb/test/'

    with open(out_file, 'w') as outfile:
        for f in onlyfiles:
            print("f",f)
            data =[{
            'boxes' : [],
            'path' : str+f}]
            yaml.dump(data, outfile, default_flow_style=False)
    print("file write done")       
            
    
    
    
if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(-1)
        
    input_folder    = sys.argv[1]
    out_file        = sys.argv[2]
    
    print ("test images",input_folder)
    tf.app.run()