
# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

bg_image = '/home/qbuser/Desktop/Screenshot from 2018-08-06 13-04-34.png'
# map_img = mpimg.imread(bg_image)


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

    def getFrame(self, sec, cap):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = cap.read()
        return hasFrames

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time - start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        boxes_center = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1] * im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3] * im_width))
            boxes_center[i] = (int((int(boxes[0, i, 1] * im_width) + int(boxes[0, i, 3] * im_width)) / 2),
                               int((int(boxes[0, i, 0] * im_height) + int(boxes[0, i, 2] * im_height)) / 2))

        # import pdb
        # pdb.set_trace()

        return boxes_list, boxes_center, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


if __name__ == "__main__":
    model_path = '/home/qbuser/Documents/Repos/big_data_works/office/HumanDetection-POC/qb_HumanDetection/resource/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture(
        '/home/qbuser/Documents/Repos/big_data_works/office/HumanDetection-POC/qb_HumanDetection/resource/TownCentreXVID.avi')
    cap.set(cv2.CAP_PROP_FPS, 60)
    count = 0
    framerate = 2
    sec = 0
    success = odapi.getFrame(sec, cap)
    all_centers = []
    while success:
        count += 1
        print(count)
        sec = sec + framerate
        sec = round(sec, 2)
        success = odapi.getFrame(sec, cap)
        r, img = cap.read()

        # img = cv2.imread(filename='/home/qbuser/Desktop/coffee_2215906b.jpg')
        img = cv2.resize(img, (1280, 720))
        boxes, centers, scores, classes, num = odapi.processFrame(img)
        previous_center = ()
        # Visualization of the results of a detection.
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img, (box[1], box[0]),
                              (box[3], box[2]), (255, 0, 0), 2)
        for i in range(len(centers)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                center = centers[i]
                all_centers.append(center)

        if sec == 10:
            points_array = np.asarray(all_centers)
            df = pd.DataFrame(points_array, columns=['x', 'y'])
            joint_kws = dict(gridsize=25)
            sns_plot = sns.jointplot('x', 'y', data=df, kind="hex",
                                     color="#4CB391", joint_kws=joint_kws)
            sns_plot.fig.axes[0].invert_yaxis()
            # plt.imshow(map_img, zorder=1, extent=[0.0, 4.8, 0.0, 6.4])
            plt.show()

        # cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
