######## Video Object Detection Using Tensorflow-trained Classifier #########
# Import packages
import os
import numpy as np
import tensorflow as tf
import sys
import cv2


sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'final1.mp4'# add your video file

# Current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector
NUM_CLASSES = 1

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
fps= video.get(cv2.CAP_PROP_FPS)
print(fps)
codec = cv2.VideoWriter_fourcc('X','V','I','D')
framerate = 30
resolution = (640,480)
VideoFileOutput= cv2.VideoWriter('this.avi',codec,framerate,resolution)

while(video.isOpened()):
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)
    VideoFileOutput.write(frame)
    
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.98)
    
    cv2.imshow('Object detector', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
VideoFileOutput.release() 
cv2.destroyAllWindows()
