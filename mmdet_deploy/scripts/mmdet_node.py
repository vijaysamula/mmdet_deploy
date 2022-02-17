#!/usr/bin/env python3
'''
This is a inference node where the detections are infered from the image.
'''


import time
from pathlib import Path
import numpy as np
import ros_numpy
import rospy
import torch
from darknet_ros_msgs.msg import BoundingBoxes
from darknet_ros_msgs.srv import CheckForObjects, CheckForObjectsResponse
from mmdet.apis import init_detector
from sensor_msgs.msg import Image

from NetTorch import inference_detector
from utils import show_result


class InferenceRos:
    def __init__(self):
        # self.cfg_path =
        rospy.init_node('mmdetection', anonymous=True)
        self.cfg_path = rospy.get_param("~cfg_file")
        self.model_path = rospy.get_param("~model_path")
        self.label_path = rospy.get_param("~label_path")
        self.score_thresh = rospy.get_param("~threshold_score")
        self.detection_name = rospy.get_param("~detection_name")
        #self.input_image_topic = rospy.get_param("input_image_topic")
        self.device = rospy.get_param("~device")
        # self.device = None
        # if (torch.cuda.is_available()):
        #     self.device = "cuda:0"
        # else :
        #     self.device = "cpu"k
        print("check whether the cuda is available :", torch.cuda.is_available())
        self.modelInference = init_detector(
            self.cfg_path,  self.model_path, device=self.device)
        print("Model is loaded succesfully")
        self.pub_arr_bbox = rospy.Publisher(
            'detected_objects', BoundingBoxes, queue_size=10)
        self.detectionImagePublisher = rospy.Publisher(
            "detection_image", Image, queue_size=10)
        #self.maskPublisher = rospy.Publisher("mask", Image ,queue_size=10)
        rospy.Subscriber("input_image_topic", Image, self.subscriberCallback)
        self.checkForObjectsService = rospy.Service(
            "check_for_objects", CheckForObjects, self.serviceCallback, buff_size=190)
        rospy.spin()

    def subscriberCallback(self, ros_msg):
        pre_t = time.time()
        rgb_img = np.frombuffer(ros_msg.data, dtype=np.uint8).reshape(
            ros_msg.height, ros_msg.width, -1)
        predictions = inference_detector(self.modelInference, rgb_img)

        detected_image, arr_bbox = show_result(
            rgb_img, predictions, self.label_path, self.detection_name, score_thr=self.score_thresh)

        detected_img_msg = ros_numpy.msgify(
            Image, detected_image, encoding='rgb8')
        # mask_msg = ros_numpy.msgify(Image,mask.astype(np.uint8), encoding='mono8')
        self.detectionImagePublisher.publish(detected_img_msg)
        # self.maskPublisher.publish(mask_msg)
        print("total callback time: ", time.time() - pre_t)
        arr_bbox.header.frame_id = ros_msg.header.frame_id
        arr_bbox.header.stamp = ros_msg.header.stamp
        if len(arr_bbox.bounding_boxes) != 0:
            self.pub_arr_bbox.publish(arr_bbox)
        torch.cuda.empty_cache()

    def serviceCallback(self, req):
        pre_t = time.time()
        img_msg = np.frombuffer(req.image.data, dtype=np.uint8).reshape(
            req.image.height, req.image.width, -1)
        predictions = inference_detector(self.modelInference, img_msg)
        detected_image, arr_bbox = show_result(
            img_msg, predictions, self.label_path, self.detection_name, score_thr=self.score_thresh)
        print("total callback time: ", time.time() - pre_t)
        arr_bbox.header.frame_id = req.image.header.frame_id
        arr_bbox.header.stamp = req.image.header.stamp
        # res.id =req.id
        # res.bounding_boxes=arr_bbox
        torch.cuda.empty_cache()
        return CheckForObjectsResponse(req.id, arr_bbox)


if __name__ == "__main__":

    InferenceRos()
