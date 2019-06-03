# Carton-box-detector-in-a-video-using-Tensorflow-object-detection-API
This model will detect carton boxes in a video(like from CCTV camera etc),and count them .This is a application in used during supply chain management and logistics.

We use tensorflow object detection API to build a custom object recognition model.

# Pre requisites
Prerequisite packages:

pillow	
lxml
jupyter	
matplotlib	
opencv

All the above packages can be installed by "pip install command"

#LabelImg
Downloading labelImg:
follow this link: https://github.com/tzutalin/labelImg
After this we will be able to get images with there .xml file that stores the bounding box values

# Downloading the TensorFlow Models
Clone this github repository and save :https://github.com/tensorflow/models

# Creating TF records:
There are two steps in doing so:

Converting the individual *.xml files to a unified *.csv file for each dataset

1.Create a new file with name xml_to_csv.py
Run the following commands:

# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv

Converting the *.csv files of each dataset to *.record files (TFRecord format)

1.Create a new file with name generate_tfrecord.py
Run the following commands:
# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/train  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/test
--output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record

# Configuring a Training Pipeline
The model we shall be using in our examples is the ssd_inception_v2_coco model.
Goto https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models
and we can choose our own model based on our performance and speed.

Download the .config file.
We need to change neccessary parameters to make model effecient.
1.Choose good learning rate.
2.Choose efficient optimizer.
3.Try and change hyperparameters
etc
Run the following command to start the training:
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config

According to the results we get , we try to trade off with other parameters and hyperparameters.

Once the training process has been initiated, you should see a series of print outs similar to the one below:
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
INFO:tensorflow:Restoring parameters from ssd_inception_v2_coco_2017_11_17/model.ckpt
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Starting Session.
INFO:tensorflow:Saving checkpoint to path training\model.ckpt
INFO:tensorflow:Starting Queues.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:global step 1: loss = 13.8886 (12.339 sec/step)
INFO:tensorflow:global step 2: loss = 16.2202 (0.937 sec/step)
INFO:tensorflow:global step 3: loss = 13.7876 (0.904 sec/step)
INFO:tensorflow:global step 4: loss = 12.9230 (0.894 sec/step)
INFO:tensorflow:global step 5: loss = 12.7497 (0.922 sec/step)
INFO:tensorflow:global step 6: loss = 11.7563 (0.936 sec/step)
INFO:tensorflow:global step 7: loss = 11.7245 (0.910 sec/step)
INFO:tensorflow:global step 8: loss = 10.7993 (0.916 sec/step)
INFO:tensorflow:global step 9: loss = 9.1277 (0.890 sec/step)
INFO:tensorflow:global step 10: loss = 9.3972 (0.919 sec/step)

# Monitor Training Job Progress using TensorBoard

To start a new TensorBoard server, we follow the following steps:
type this in your cmd prompt

activate tensorflow_gpu

cd into the training_demo folder.

Run the following command:

tensorboard --logdir=training\

We can visualize the graphs in :http://YOUR-PC:6006 or http://localhost:6006













