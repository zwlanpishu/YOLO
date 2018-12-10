import numpy as np 
import tensorflow as tf 
import scipy
import h5py
import math
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


"""
以下代码仅为从YOLO算法所有的输出数据中，这里以(19, 19, 5, 85)为例，进行过滤
处理，最后输出指定的有效数据的代码。YOLO算法的初始输出数据可以认为是在19x19
的栅格里存5个box，每个box有85个检测数据。但是这些数据并不是全部都有效的，需
要对其进行过滤处理最后，只输出若干个有效的输出数据，用来表征bounding box的相关信息。
"""

"""
box_confidence -- tensor of shape (19, 19, 5, 1)
boxes -- tensor of shape (19, 19, 5, 4)
box_class_probs -- tensor of shape (19, 19, 5, 80)

This func is used to filter the valid boxes from all boxes predicted
"""
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.6) : 
    
    # the shape of box_scores is (19, 19, 5, 80)
    # while for each box, there are 80 classes to be detected
    box_scores = box_confidence * box_class_probs

    # for each box, find the most probably class
    box_classes = tf.arg_max(box_scores, -1)

    # for each box, find the max score
    box_class_scores = tf.reduce_max(box_scores, -1)
    
    # generate the filtering mask
    filter_mask = box_class_scores > threshold

    # the final boxes which is valid
    scores = tf.boolean_mask(box_class_scores, filter_mask)
    boxes_bounding = tf.boolean_mask(boxes, filter_mask)
    classes = tf.boolean_mask(box_classes, filter_mask)

    return  scores, boxes_bounding, classes


"""
This func is used to implement the intersection over union (IoU) between box1 and box2
"""
def iou(box1, box2) : 

    # get the coordinate of upper left and lower right of box1 
    x1_box1 = box1[0]
    y1_box1 = box1[1]
    x2_box1 = box1[2]
    y2_box1 = box1[3]

    # get the coordinate of upper left and lower right of box2
    x1_box2 = box2[0]
    y1_box2 = box2[1]
    x2_box2 = box2[2]
    y2_box2 = box2[3]

    # the coordinate of intersection
    x1_i = max(x1_box1, x1_box2)
    y1_i = max(y1_box1, y1_box2)
    x2_i = min(x2_box1, x2_box2)
    y2_i = min(y2_box1, y2_box2)

    # get the inter_area of box1 and box2
    inter_area = max(y2_i - y1_i, 0) * max(x2_i - x1_i, 0)

    # get the union-area of box1 and box2
    box1_area = (y2_box1 - y1_box1) * (x2_box1 - x1_box1)
    box2_area = (y2_box2 - y1_box2) * (x2_box2 - x1_box2)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou


"""
Convert YOLO box predictions to bounding box corners.
"""
def yolo_boxes_to_corners(box_xy, box_wh):
    
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.concat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2], # y_max
        box_maxes[..., 0:1]  # x_max
        ], -1)


"""
Scales the predicted boxes in order to be drawable on the image
"""
def scale_boxes(boxes, image_shape) :
    height = image_shape[0]
    width = image_shape[1]
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes


"""
This func is used to implement the non-max-suppression of all boxes
Arguments:
    scores -- tensor of shape (None,) 
    boxes_bounding -- tensor of shape (None, 4)
    classes -- tensor of shape (None,)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
"""
def yolo_non_max_suppression(scores, boxes_bounding, classes, max_boxes = 10, iou_threshold = 0.5) : 

    # non_max_suppression can be implemented by the build-in func of tensorflow
    # the return of this func is the index of boxes_bounding which satisfy the iou condition
    select_indices = tf.image.non_max_suppression(boxes_bounding, scores, max_boxes, iou_threshold)
    select_scores = tf.gather(scores, select_indices)
    select_boxes_bounding = tf.gather(boxes_bounding, select_indices)
    select_classes = tf.gather(classes, select_indices)

    return select_scores, select_boxes_bounding, select_classes


"""
This func is used to convert the original yolo output to a filtering result
Argument:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3))
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value
    iou_threshold -- real value
"""
def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes = 10, 
              score_threshold = 0.6, iou_threshold = 0.5) :
    
    box_confidence, boxes_xy, boxes_wh, box_class_probs = yolo_outputs
    
    # convert the one form of boxes to another, the shape of boxes is (19, 19, 5, 4)
    boxes = yolo_boxes_to_corners(boxes_xy, boxes_wh)
    scores, boxes_bounding, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, 
                                                        threshold = score_threshold)
    
    # scale the bounding of box to original image shape
    boxes_bounding = scale_boxes(boxes_bounding, image_shape)

    # filter the result by non_max_suppression
    scores, boxes_bounding, classes = yolo_non_max_suppression(scores, boxes_bounding, classes,
                                                               max_boxes = max_boxes, 
                                                               iou_threshold = iou_threshold)

    return scores, boxes_bounding, classes








