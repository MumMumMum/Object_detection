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
Provide path dir over here:
flags.DEFINE_string('input_path', 'additional_train.yaml', 'Path to the yaml input')
flags.DEFINE_string('output_path', 'train_TFRecord', 'Path to output TFRecord')
change these 'additional_train.yaml','train_TFRecord'

object_detection_tutorial.ipynb is the script where we test our images.(Load the model and run test.)

Bosch data
https://hci.iwr.uni-heidelberg.de/node/6132/download/8fec5eefe8aea975e15ecda1eec6fc0e
We need to just take rgb  images and not riib images

Added script generate_yaml.py. This is used to read all iamges in test dir and generate yaml file.Then from .yaml we need to get TF_record for test dir.


The dir for Bosch data where we keep script to generate yaml,TF record looks like this
![DIR_STRUCT](/img/img.jpg?raw=true "Optional Title")



>Data dir struct for tainining model:
>
>+data
>  -label_map file
>  -train TFRecord file
>  -eval TFRecord file
>+models
>  + model
>    -pipeline config file
>   +train
>   +eval



