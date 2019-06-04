# Carton-box-detector-in-a-video-using-Tensorflow-object-detection-API
This model will detect carton boxes in a video(like from CCTV camera etc),and count them .This is a application in used during supply chain management and logistics.

We use tensorflow object detection API to build a custom object recognition model.

# Pre requisites
Prerequisite packages:

pillow, 	
lxml, 
jupyter, 
matplotlib, 
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
*python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
*python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv

Converting the *.csv files of each dataset to *.record files (TFRecord format)




1.Create a new file with name generate_tfrecord.py
Run the following commands:

python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv
--img_path=<PATH_TO_IMAGES_FOLDER>/train  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

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

Total loss after :global step 20: loss = 5.4179 (77.200 sec/step)
# Monitor Training Job Progress using TensorBoard

To start a new TensorBoard server, we follow the following steps:
type this in your cmd prompt:

activate tensorflow_gpu

cd into the training_demo folder.

Run the following command:

tensorboard --logdir=training\

We can visualize the graphs in :http://YOUR-PC:6006 or http://localhost:6006

Finally in order to test the model on live video use webcam_demo.py file (This is a general object detection code) 
and change model to your model and run the code.









