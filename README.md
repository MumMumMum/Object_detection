## Bosch Small Traffic Lights Dataset

Installation :
We need this API downloaded and installed, https://github.com/tensorflow/model
Also all these will be need only to train model, 
on Carla we dont need these we are just going to use freezed model.


Refer this to complete the installation:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

After installation,we need to follow these steps mentioned below:
(it is not mentioned but we need this for train.py)
setup.py build
setup.py install
set PYTHONPATH=G:\CarND\ObjectDetection\models-master;G:\CarND\ObjectDetection\models-master\research\slim(windows)
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim(linux)

dataset_stats.py .... gives all the classes and nos of images per class
Used in generatin label map pb.txt

generate_tfrecord.py input tp pretrained models.

object_detection_tutorial.ipynb is the script where we test our images.(Load the model and run test.)



