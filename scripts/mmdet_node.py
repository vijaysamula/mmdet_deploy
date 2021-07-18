#!/opt/conda/bin/python



import rospy
import rospkg
import numpy as np
import sys
import os
import time
import warnings

import ros_numpy
import cv2
import torch 
import glob
from pathlib import Path
from cv_bridge import CvBridge, CvBridgeError

from mmdet.apis import init_detector
from NetTorch import inference_detector
from sensor_msgs.msg import Image
from mmdet_deploy.msg import BoundingBoxInstance,BoundingBoxesInstances,ObjectCount
from mmdet_deploy.srv import CheckForObjects, CheckForObjectsResponse



from utils import show_result

class InferenceRos:
    def __init__(self):
        #self.cfg_path = 
        rospy.init_node('mmdetection', anonymous=True)
        self.cfg_path = rospy.get_param("~cfg_file")
        self.model_path = rospy.get_param("~model_path")
        self.label_path = rospy.get_param("~label_path")
        self.score_thresh = rospy.get_param("~threshold_score")
        self.detection_name= rospy.get_param("~detection_name")
        self.input_image_topic = rospy.get_param("~input_image_topic")
        self.device = None
        if (torch.cuda.is_available()):
            self.device = "cuda:2"
        else :
            self.device = "cpu"

        self.modelInference = init_detector(self.cfg_path,  self.model_path , device=self.device)
        print("Model is loaded succesfully")
        self.pub_arr_bbox = rospy.Publisher('detected_objects', BoundingBoxesInstances, queue_size=10)
        self.detectionImagePublisher = rospy.Publisher("detection_image", Image ,queue_size=10)
        #self.maskPublisher = rospy.Publisher("mask", Image ,queue_size=10)
        rospy.Subscriber(self.input_image_topic, Image, self.subscriberCallback)
        self.checkForObjectsService = rospy.Service("check_for_2D_objects",CheckForObjects,self.serviceCallback)
        rospy.spin()
    
    def subscriberCallback(self,ros_msg):
        pre_t = time.time()
        rgb_img =  np.frombuffer(ros_msg.data, dtype=np.uint8).reshape(ros_msg.height, ros_msg.width, -1)
        predictions = inference_detector(self.modelInference, rgb_img)
        
        detected_image,arr_bbox = show_result(rgb_img, predictions,self.label_path,self.detection_name ,score_thr=self.score_thresh)
        
        detected_img_msg = ros_numpy.msgify(Image,detected_image, encoding='rgb8')
        # mask_msg = ros_numpy.msgify(Image,mask.astype(np.uint8), encoding='mono8')
        self.detectionImagePublisher.publish(detected_img_msg) 
        #self.maskPublisher.publish(mask_msg)
        print("total callback time: ", time.time() - pre_t)
        arr_bbox.header.frame_id = ros_msg.header.frame_id
        arr_bbox.header.stamp = ros_msg.header.stamp
        #print(arr_bbox)
        if len(arr_bbox.bounding_boxes) != 0:
            self.pub_arr_bbox.publish(arr_bbox)
            

        
        
    def serviceCallback(self,req):
        pre_t = time.time()
        
        
        img_msg = np.frombuffer(req.image.data, dtype=np.uint8).reshape(req.image.height, req.image.width, -1)
        print("  ")
        predictions = inference_detector(self.modelInference, img_msg)
        
        detected_image,arr_bbox = self.modelInference.show_result(img_msg, predictions, score_thr=self.score_thresh)
        
        
        self.detectionImagePublisher.publish(detected_image)       
        print("total callback time: ", time.time() - pre_t)
        arr_bbox.header.frame_id = img_msg.header.frame_id
        arr_bbox.header.stamp = img_msg.header.stamp
        
        return CheckForObjectsResponse(arr_bbox)
            


if __name__ == "__main__":
    
    
    InferenceRos()
    rospy.spin()

