# object_detection.py

# Import libraries
import numpy as np
import cv2
from ultralytics import YOLO
import json
import yaml


class ObjectDetector:
    def __init__(self, json_path="Detection/config/pitch map labels position.json",
                 field_yaml_path="config/config pitch dataset.yaml", players_yaml_path="config/config players dataset.yaml",
                 tac_map_path="assets/tactical map.jpg", model_keypoints_path="models/Field_Keypoints.pt",
                 model_players_path="models/Players.pt", player_model_conf_thresh=0.60,
                 keypoints_model_conf_thresh=0.70, nbr_frames_no_ball_thresh=30, ball_track_dist_thresh=100):
        self.json_path = json_path
        self.field_yaml_path = field_yaml_path
        self.players_yaml_path = players_yaml_path
        self.tac_map_path = tac_map_path
        self.model_keypoints_path = model_keypoints_path
        self.model_players_path = model_players_path
        self.player_model_conf_thresh = player_model_conf_thresh
        self.keypoints_model_conf_thresh = keypoints_model_conf_thresh
        self.nbr_frames_no_ball_thresh = nbr_frames_no_ball_thresh
        self.ball_track_dist_thresh = ball_track_dist_thresh

    def load(self):
        # Get tactical map keypoints positions dictionary
        with open(self.json_path, 'r') as f:
            self.keypoints_map_pos = json.load(f)
        # Get football field keypoints numerical to alphabetical mapping
        with open(self.field_yaml_path, 'r') as file:
            classes_names_dic = yaml.safe_load(file)
        self.classes_names_dic = classes_names_dic['names']
        # Get players mapping
        with open(self.players_yaml_path, 'r') as file:
            labels_dic = yaml.safe_load(file)
        labels_dic = labels_dic['names']
        # Read tactical map image
        self.tac_map = cv2.imread(self.tac_map_path)
        # Load the YOLOv8 players detection model
        self.model_players = YOLO(self.model_players_path)
        # Load the YOLOv8 field keypoints detection model
        self.model_keypoints = YOLO(self.model_keypoints_path)

        return labels_dic

    def detect_players(self, frame):

        # Run YOLOv8 players inference on the frame
        results_players = self.model_players(frame, conf=self.player_model_conf_thresh)

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

    def detect_keypoints(self, frame):

        # Run YOLOv8 field keypoints inference on the frame
        results_keypoints = self.model_keypoints(frame, conf=self.keypoints_model_conf_thresh)

        ## Extract detections information
        bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()  # Detected field keypoints (x,y,w,h) bounding boxes
        bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()  # Detected field keypoints (x,y,w,h) bounding boxes
        labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())  # Detected field keypoints labels list

        # Convert detected numerical labels to alphabetical labels
        detected_labels = [self.classes_names_dic[i] for i in labels_k]

        # Extract detected field keypoints coordiantes on the current frame
        detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

        # Get the detected field keypoints coordinates on the tactical map
        detected_labels_dst_pts = np.array([self.keypoints_map_pos[i] for i in detected_labels])

        return bboxes_k, detected_labels, detected_labels_src_pts, detected_labels_dst_pts