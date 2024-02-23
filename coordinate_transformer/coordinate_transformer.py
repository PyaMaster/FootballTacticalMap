# coordinate_transformation.py

# Import libraries
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error

def calculate_homography(frame_nbr, detected_labels, detected_labels_src_pts, detected_labels_dst_pts,
                         detected_labels_prev, detected_labels_src_pts_prev, keypoints_displacement_mean_tol):
    # Implement homography calculation logic here

    # Always calculate homography matrix on the first frame
    if frame_nbr > 1:
        # Determine common detected field keypoints between previous and current frames
        common_labels = set(detected_labels_prev) & set(detected_labels)
        # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
        if len(common_labels) > 3:
            common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]  # Get labels indexes of common detected keypoints from previous frame
            common_label_idx_curr = [detected_labels.index(i) for i in common_labels]  # Get labels indexes of common detected keypoints from current frame
            coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]  # Get labels coordiantes of common detected keypoints from previous frame
            coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]  # Get labels coordiantes of common detected keypoints from current frame
            coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  # Calculate error between previous and current common keypoints coordinates
            update_homography = coor_error > keypoints_displacement_mean_tol  # Check if error surpassed the predefined tolerance level
        else:
            update_homography = True
    else:
        update_homography = True

    if update_homography:
        h, mask = cv2.findHomography(detected_labels_src_pts, detected_labels_dst_pts) # Calculate homography matrix
    else:
        h = np.array([])
        mask = None

    detected_labels_prev = detected_labels.copy()  # Save current detected keypoint labels for next frame
    detected_labels_src_pts_prev = detected_labels_src_pts.copy()  # Save current detected keypoint coordiantes for next frame

    return h, mask, detected_labels_prev, detected_labels_src_pts_prev


def transform_coordinates(h, detected_ppos_src_pts, detected_ball_src_pos, ball_track_history, ball_track_dist_thresh, max_track_length, show_b):
    # Implement coordinate transformation logic here

    # Transform players coordinates from frame plan to tactical map plance using the calculated Homography matrix
    pred_dst_pts = []  # Initialize players tactical map coordiantes list
    for pt in detected_ppos_src_pts:  # Loop over players frame coordiantes
        pt = np.append(np.array(pt), np.array([1]), axis=0)  # Covert to homogeneous coordiantes
        dest_point = np.matmul(h, np.transpose(pt))  # Apply homography transofrmation
        dest_point = dest_point / dest_point[2]  # Revert to 2D-coordiantes
        pred_dst_pts.append(list(np.transpose(dest_point)[:2]))  # Update players tactical map coordiantes list
    pred_dst_pts = np.array(pred_dst_pts)

    # Transform ball coordinates from frame plan to tactical map plane using the calculated Homography matrix
    if detected_ball_src_pos is not None:
        pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
        dest_point = np.matmul(h, np.transpose(pt))
        dest_point = dest_point / dest_point[2]
        detected_ball_dst_pos = np.transpose(dest_point)

        # Update track ball position history
        if show_b:
            if len(ball_track_history['src']) > 0:
                if np.linalg.norm(detected_ball_src_pos - ball_track_history['src'][-1]) < ball_track_dist_thresh:
                    ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                    ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                else:
                    ball_track_history['src'] = [(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                    ball_track_history['dst'] = [(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
            else:
                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
    # Remove oldest tracked ball postion if track exceedes threshold
    if len(ball_track_history) > max_track_length:
        ball_track_history['src'].pop(0)
        ball_track_history['dst'].pop(0)

    return pred_dst_pts, detected_ball_dst_pos, ball_track_history