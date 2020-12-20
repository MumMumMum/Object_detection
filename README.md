## Bosch Small Traffic Lights Dataset  

**Installation :**
We need this API downloaded and installed, https://github.com/tensorflow/models Also all these will be need only to train model, 
on Carla we dont need these we will need just classify.  

Refer this to complete the installation:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md  


After installation,we need to follow these steps mentioned below:  
(it is not mentioned but we need this for train.py)  
1.setup.py build  
2.setup.py install  
3. windows:`set PYTHONPATH=G:\CarND\ObjectDetection\models-master;G:\CarND\ObjectDetection\models-master\research\slim`    
4. linux : ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim ```  

### FILES to genererate TFRecord.record   
1. *dataset_stats.py*: gives all the classes and nos of images per class     
      *Used in generatin label map pb.txt  

2. *generate_tfrecord.py*: This files reads yaml files and takes iamges and generates the TFrecord files.
      *the filename args, are passed in flags over here.  
         flags.DEFINE_string('input_path', 'additional_train.yaml', 'Path to the yaml input')  
         flags.DEFINE_string('output_path', 'train_TFRecord', 'Path to output TFRecord')  
      

3. *object_detection_tutorial.ipynb* is the script where we test our images.(Load the model and run test.)

### Bosch data  
https://hci.iwr.uni-heidelberg.de/node/6132/download/8fec5eefe8aea975e15ecda1eec6fc0e
- We need to just take rgb  images and not riib images   
- Test Data has no yaml  
- use generate_yaml.py generate yaml on test Data.  

### Config files  
    Need to edit ssd_mobilenet_v1_coco.config for the model.  
    - Nos of classes 14.  
    - Path of Test and Train record. I have provided full path, Linux user can give relative path.  
    
    
### Data Dir
  For training we need to work on Object detection setup.The dir struct looks like this:  
  ```
  +data  
      -label_map file  
      -train TFRecord file  
      -eval TFRecord file  
  +models  
    + model  
      -pipeline config file  
      +train  
      +eval  
  ```


### Training  
  For train.py use command :
  ```
      python train.py --logtostderr --pipeline_config_path=G:\CarND\ObjectDetection\models-master\research\object_detection \models\ssd_mobilenet_v1_coco_11_06_2017\ssd_mobilenet_v1_coco.config --train_dir=G:\CarND\ObjectDetection\models-master\research\object_detection\data
```
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md 




