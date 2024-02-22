# object_detection.py

# Import libraries
import numpy as np
import cv2
from ultralytics import YOLO

def detect_players(frame, model_players, player_model_conf_thresh):

    # Run YOLOv8 players inference on the frame
    results_players = model_players(frame, conf=player_model_conf_thresh)

    ## Extract detections information
    bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()  # Detected players, referees and ball (x,y,x,y) bounding boxes
    bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()  # Detected players, referees and ball (x,y,w,h) bounding boxes
    labels_p = list(results_players[0].boxes.cls.cpu().numpy())  # Detected players, referees and ball labels list
    confs_p = list(results_players[0].boxes.conf.cpu().numpy())  # Detected players, referees and ball confidence level

    bboxes_p_c_0 = bboxes_p_c[[i == 0 for i in labels_p],:]  # Get bounding boxes information (x,y,w,h) of detected players (label 0)
    bboxes_p_c_2 = bboxes_p_c[[i == 2 for i in labels_p],:]  # Get bounding boxes information (x,y,w,h) of detected ball(s) (label 2)

    # Get coordinates of detected players on frame (x_cencter, y_center+h/2)
    detected_ppos_src_pts = bboxes_p_c_0[:, :2] + np.array([[0] * bboxes_p_c_0.shape[0], bboxes_p_c_0[:, 3]/2]).transpose()
    # Get coordinates of the first detected ball (x_center, y_center)
    detected_ball_src_pos = bboxes_p_c_2[0, :2] if bboxes_p_c_2.shape[0] > 0 else None

    return results_players, bboxes_p, labels_p, confs_p, detected_ppos_src_pts, detected_ball_src_pos

def detect_keypoints(frame, model_keypoints, keypoints_model_conf_thresh, classes_names_dic, keypoints_map_pos):

    # Run YOLOv8 field keypoints inference on the frame
    results_keypoints = model_keypoints(frame, conf=keypoints_model_conf_thresh)

    ## Extract detections information
    bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()  # Detected field keypoints (x,y,w,h) bounding boxes
    bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()  # Detected field keypoints (x,y,w,h) bounding boxes
    labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())  # Detected field keypoints labels list

    # Convert detected numerical labels to alphabetical labels
    detected_labels = [classes_names_dic[i] for i in labels_k]

    # Extract detected field keypoints coordiantes on the current frame
    detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

    # Get the detected field keypoints coordinates on the tactical map
    detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])

    return bboxes_k, detected_labels, detected_labels_src_pts, detected_labels_dst_pts