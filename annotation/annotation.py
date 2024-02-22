# annotation.py

# Import libraries
import numpy as np
import pandas as pd

import cv2
import skimage
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error

import json
import yaml
import time

def annotate_frame(annotated_frame, bboxes_p, confs_p, labels_p, obj_palette_list, colors_dic, players_teams_list,
                   detected_labels_src_pts, pred_dst_pts, labels_dic, detected_ball_src_pos, detected_ball_dst_pos, bboxes_k):
    # Implement annotation logic here

    ball_color_bgr = (0, 0, 255)  # Color (GBR) for ball annotation on tactical map
    j = 0  # Initializing counter of detected players
    palette_box_size = 10  # Set color box size in pixels (for display)

    # Loop over all detected object by players detection model
    for i in range(bboxes_p.shape[0]):
        conf = confs_p[i]  # Get confidence of current detected object
        if labels_p[i] == 0:  # Display annotation for detected players (label 0)

            # Display extracted color palette for each detected player
            palette = obj_palette_list[j]  # Get color palette of the detected player
            for k, c in enumerate(palette):
                c_bgr = c[::-1]  # Convert color to BGR
                annotated_frame = cv2.rectangle(annotated_frame,
                                                (int(bboxes_p[i, 2]) + 3,  # Add color palette annotation on frame
                                                 int(bboxes_p[i, 1]) + k * palette_box_size),
                                                (int(bboxes_p[i, 2]) + palette_box_size,
                                                 int(bboxes_p[i, 1]) + (palette_box_size) * (k + 1)),
                                                c_bgr, -1)

            team_name = list(colors_dic.keys())[players_teams_list[j]]  # Get detected player team prediction
            color_rgb = colors_dic[team_name][0]  # Get detected player team color
            color_bgr = color_rgb[::-1]  # Convert color to bgr

            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 0]), int(bboxes_p[i, 1])),
                                            # Add bbox annotations with team colors
                                            (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])), color_bgr, 1)

            cv2.putText(annotated_frame, team_name + f" {conf:.2f}",  # Add team name annotations
                        (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color_bgr, 2)

            # Add tactical map player postion color coded annotation if more than 3 field keypoints are detected
            if len(detected_labels_src_pts) > 3:
                tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])),
                                          radius=5, color=color_bgr, thickness=-1)

            j += 1  # Update players counter
        else:  # Display annotation for otehr detections (label 1, 2)
            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 0]), int(bboxes_p[i, 1])),
                                            # Add white colored bbox annotations
                                            (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])), (255, 255, 255), 1)
            cv2.putText(annotated_frame, labels_dic[labels_p[i]] + f" {conf:.2f}",
                        # Add white colored label text annotations
                        (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

            # Add tactical map ball postion annotation if detected
            if detected_ball_src_pos is not None:
                tac_map_copy = cv2.circle(tac_map_copy, (int(detected_ball_dst_pos[0]),
                                                         int(detected_ball_dst_pos[1])), radius=5,
                                          color=ball_color_bgr, thickness=3)
    for i in range(bboxes_k.shape[0]):
        annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i, 0]), int(bboxes_k[i, 1])),
                                        # Add bbox annotations with team colors
                                        (int(bboxes_k[i, 2]), int(bboxes_k[i, 3])), (0, 0, 0), 1)

    return annotated_frame

def annotate_tactical_map(tac_map_copy, annotated_frame, ball_track_history, prev_frame_time):
    # Implement tactical map annotation logic here

    # Plot the ball tracks on tactical map
    if len(ball_track_history['src']) > 0:
        points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
        tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2)

    # Combine annotated frame and tactical map in one image with colored border separation
    border_color = [255, 255, 255]  # Set border color (BGR)
    annotated_frame = cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10,  # Add borders to annotated frame
                                         cv2.BORDER_CONSTANT, value=border_color)
    tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT,
                                      # Add borders to tactical map
                                      value=border_color)
    tac_map_copy = cv2.resize(tac_map_copy,
                              (tac_map_copy.shape[1], annotated_frame.shape[0]))  # Resize tactical map
    final_img = cv2.hconcat((annotated_frame, tac_map_copy))  # Concatenate both images
    ## Add info annotation
    cv2.putText(final_img, "Tactical Map", (1370, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(final_img, "Press 'p' to pause & 'q' to quit", (820, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0),
                2)

    new_frame_time = time.time()  # Get time after finished processing current frame
    fps = 1 / (new_frame_time - prev_frame_time)  # Calculate FPS as 1/(frame proceesing duration)
    prev_frame_time = new_frame_time  # Save current time to be used in next frame
    cv2.putText(final_img, "FPS: " + str(int(fps)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return prev_frame_time, final_img