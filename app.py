import tempfile
import numpy as np

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO

import os
import json
import yaml
import time

import Detection.detection as dt
import team_prediction.team_prediction as tp
import coordinate_transformer.coordinate_transformer as ct
import annotation.annotation as at

def main():
    st.set_page_config(page_title="AI Powered Web Application for Football Tactical Analysis", layout="wide", initial_sidebar_state="expanded")
    st.title("Football Players Detection With Team Prediction & Tactical Map")
    st.subheader(":red[Works only with Tactical Camera footage]")

    st.sidebar.title("Main Settings")

    ## Sidebar Setup
    st.sidebar.markdown("---")
    st.sidebar.subheader("Video upload")
    input_video_file = st.sidebar.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi', 'm4v', 'asf'])

    demo_video_path = "assets/test vid.mp4"
    demo_team_info = {
        "team1_name": "Chelsea",
        "team2_name": "Manchester City"
    }

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if not input_video_file:
        tempf.name = demo_video_path
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Demo video')
        st.sidebar.video(demo_bytes)
    else:
        tempf.write(input_video_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)

    # Get tactical map keypoints positions dictionary
    json_path = 'config/pitch map labels position.json'
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)
    # Get football field keypoints numerical to alphabetical mapping
    field_yaml_path = 'config/config pitch dataset.yaml'
    with open(field_yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    classes_names_dic = classes_names_dic['names']
    # Get players mapping
    players_yaml_path = 'config/config players dataset.yaml'
    with open(players_yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    # Read tactical map image
    tac_map_path = 'assets/tactical map.jpg'
    tac_map = cv2.imread(tac_map_path)
    # Load the YOLOv8 players detection model
    model_players = YOLO("models/Players.pt")
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("models/Field_Keypoints.pt")

    st.sidebar.markdown('---')
    st.sidebar.subheader("Team Names")
    if not input_video_file:
        team1_name = st.sidebar.text_input(label='First Team Name', value=demo_team_info["team1_name"])
        team2_name = st.sidebar.text_input(label='Second Team Name', value=demo_team_info["team2_name"])
    else:
        team1_name = st.sidebar.text_input(label='First Team Name')
        team2_name = st.sidebar.text_input(label='Second Team Name')
    st.sidebar.markdown('---')

    ## Page Setup
    tab1, tab2, tab3 = st.tabs(["How to use ?", "Team Colors", "Model Hyperparameters & Detection"])
    with tab1:
        st.header(':blue[Welcome!]')
        st.subheader('Main Application Functionalities:', divider='blue')
        st.markdown("""
                    1. Football players, referee, and ball detection.
                    2. Players team prediction.
                    3. Estimation of players and ball positions on a tactical map.
                    4. Ball Tracking.
                    """)
        st.subheader('How to use?', divider='blue')
        st.markdown("""
                    **There is a demo videos that are automaticaly loaded when you start the app**
                    1. Upload a video to analyse, using the sidebar menu "Browse files" button.
                    2. Enter the team names that corresponds to the uploaded video in the text fields in the sidebar menu.
                    3. Access the "Team colors" tab in the main page.
                    4. Select a frame where players and goal keepers from both teams can be detected.
                    5. Follow the instruction on the page to pick each team colors.
                    6. Go to the "Model Hyperpramerters & Detection" tab, adjust hyperparameters and select the annotation options. (Default hyperparameters are recommended)
                    7. Run Detection!
                    8. If "save outputs" option was selected the saved video can be found in the "outputs" directory
                    """)
        st.write("Version 0.0.1")

    with tab2:
        t1col1, t1col2 = st.columns([1, 1])
        with t1col1:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1,
                                  help="Select frame to pick team colors from")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            with st.spinner('Detecting players in selected frame..'):
                results = model_players(frame, conf=0.7)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections_imgs_list = []
                detections_imgs_grid = []
                padding_img = np.ones((80, 60, 3), dtype=np.uint8) * 255
                for i, j in enumerate(list(labels)):
                    if int(j) == 0:
                        bbox = bboxes[i,:]
                        obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        obj_img = cv2.resize(obj_img, (60,80))
                        detections_imgs_list.append(obj_img)
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list) // 2)])
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list) // 2, len(detections_imgs_list))])
                if len(detections_imgs_list) % 2 != 0:
                    detections_imgs_grid[0].append(padding_img)
                concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                concat_det_imgs = cv2.vconcat([concat_det_imgs_row1, concat_det_imgs_row2])
            st.write("Detected players")
            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")

            st.markdown('---')
            radio_options = [f"{team1_name} P color", f"{team1_name} GK color", f"{team2_name} P color", f"{team2_name} GK color"]
            active_color = st.radio(label="Select which team color to pick from the image above", options=radio_options, horizontal=True,
                                    help="Chose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")
            if value is not None:
                picked_color = concat_det_imgs[value['y'], value['x'], :]
                st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)
            st.write("Boxes below can be used to manually adjust selected colors.")
            cp1, cp2, cp3, cp4 = st.columns([1, 1, 1, 1])
            with cp1:
                hex_color_1 = st.session_state[f"{team1_name} P color"] if f"{team1_name} P color" in st.session_state else "#f2f5fb"
                team1_p_color = st.color_picker(label=f"{team1_name} P", value=hex_color_1, key='t1p')
                st.session_state[f"{team1_name} P color"] = team1_p_color
            with cp2:
                hex_color_2 = st.session_state[f"{team1_name} GK color"] if f"{team1_name} GK color" in st.session_state else "#f2f5fb"
                team1_gk_color = st.color_picker(label=f"{team1_name} GK", value=hex_color_2, key='t1gk')
                st.session_state[f"{team1_name} GK color"] = team1_gk_color
            with cp3:
                hex_color_3 = st.session_state[f"{team2_name} P color"] if f"{team2_name} P color" in st.session_state else "#f2f5fb"
                team2_p_color = st.color_picker(label=f"{team2_name} P", value=hex_color_3, key='t2p')
                st.session_state[f"{team2_name} P color"] = team2_p_color
            with cp4:
                hex_color_4 = st.session_state[f"{team2_name} GK color"] if f"{team2_name} GK color" in st.session_state else "#f2f5fb"
                team2_gk_color = st.color_picker(label=f"{team2_name} GK", value=hex_color_4, key='t2gk')
                st.session_state[f"{team2_name} GK color"] = team2_gk_color
        st.markdown('---')

        with t1col2:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR")

    colors_dic, color_list_lab = tp.create_colors_info(team1_name=team1_name, team1_p_color=st.session_state[f"{team1_name} P color"],
                                                                     team1_gk_color=st.session_state[ f"{team1_name} GK color"],
                                                                     team2_name=team2_name, team2_p_color=st.session_state[f"{team2_name} P color"],
                                                                     team2_gk_color=st.session_state[f"{team2_name} GK color"])

    with tab3:
        t2col1, t2col2 = st.columns([1, 1])
        with t2col1:
            player_model_conf_thresh = st.slider('PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.6)
            keypoints_model_conf_thresh = st.slider('Field Keypoints PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.7)
            keypoints_displacement_mean_tol = st.slider('Keypoints Displacement RMSE Tolerance (pixels)', min_value=-1, max_value=100, value=7,
                                                        help="Indicates the maximum allowed average distance between the position of the field keypoints\
                                                                       in current and previous detections. It is used to determine wether to update homography matrix or not. ")
        with t2col2:
            num_pal_colors = st.slider(label="Number of palette colors", min_value=1, max_value=5, step=1, value=3,
                                       help="How many colors to extract form detected players bounding-boxes? It is used for team prediction.")
            st.markdown("---")
            save_output = st.checkbox(label='Save output', value=False)
            if save_output:
                output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.')
            else:
                output_file_name = None
        st.markdown("---")

        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            nbr_frames_no_ball_thresh = st.number_input("Ball track reset threshold (frames)", min_value=1, max_value=10000, value=30,
                                                        help="After how many frames with no ball detection, should the track be reset?")
            ball_track_dist_thresh = st.number_input("Ball track distance threshold (pixels)", min_value=1, max_value=1280, value=100,
                                                     help="Maximum allowed distance between two consecutive balls detection to keep the current track.")
            max_track_length = st.number_input("Maximum ball track length (Nbr. detections)", min_value=1, max_value=1000, value=35,
                                               help="Maximum total number of ball detections to keep in tracking history")
        with bcol2:
            st.write("Annotation options:")
            bcol21t, bcol22t = st.columns([1, 1])
            with bcol21t:
                show_k = st.toggle(label="Show Keypoints Detections", value=False)
                show_p = st.toggle(label="Show Players Detections", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Show Color Palettes", value=True)
                show_b = st.toggle(label="Show Ball Tracks", value=True)
            st.markdown('---')
            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5, 1, 1, 1])
            with bcol21:
                st.write('')
            with bcol22:
                ready = True if (team1_name == '') or (team2_name == '') else False
                start_detection = st.button(label='Start Detection', disabled=ready)
            with bcol23:
                stop_btn_state = True if not start_detection else False
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state)
            with bcol24:
                st.write('')

    stframe = st.empty()
    cap = cv2.VideoCapture(tempf.name)
    status = False

    if start_detection and not stop_detection:
        st.toast(f'Detection Started !')

        nbr_team_colors = len(list(colors_dic.values())[0])

        if (output_file_name is not None) and (len(output_file_name) == 0):
            list_video_files = os.listdir('./outputs/')
            idx = 0
            while True:
                idx += 1
                output_file_name = f'detect_{idx}'
                if output_file_name + '.mp4' not in list_video_files:
                    break

        # Get Tactical map image size
        tac_width = tac_map.shape[0]
        tac_height = tac_map.shape[1]

        # Create output video writer
        if save_output:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + tac_width
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + tac_height
            output = cv2.VideoWriter(f'./outputs/{output_file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

        # Create progress bar
        tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st_prog_bar = st.progress(0, text='Detection starting.')

        # Initialize frame counter
        frame_nbr = 0
        # Set variable to record the time when we processed last frame
        prev_frame_time = 0
        # Set variable to record the time at which we processed current frame
        new_frame_time = 0
        # Count consecutive frames with no ball detected
        nbr_frames_no_ball = 0
        # Store the ball track history
        ball_track_history = {'src': [],
                              'dst': []
                              }

        # set detected_labels_prev and detected_labels_src_pts_prev
        detected_labels_prev = None
        detected_labels_src_pts_prev = None

        while cap.isOpened():

            # Update progress bar
            percent_complete = int(frame_nbr / (tot_nbr_frames) * 100)
            st_prog_bar.progress(percent_complete, text=f"Detection in progress ({percent_complete}%)")

            # Update frame counter
            frame_nbr += 1

            # Read a frame from the video
            success, frame = cap.read()

            # Reset tactical map image for each new frame
            tac_map_copy = tac_map.copy()

            # Reset ball tracks
            if nbr_frames_no_ball > nbr_frames_no_ball_thresh:
                ball_track_history['dst'] = []
                ball_track_history['src'] = []
            # Process the frame if it was successfuly read
            if success:
                ## Extract detections information for field keypoints
                bboxes_k, detected_labels, detected_labels_src_pts, detected_labels_dst_pts = dt.detect_keypoints(frame, model_keypoints, keypoints_model_conf_thresh, classes_names_dic, keypoints_map_pos)

                ## Calculate Homography transformation matrix when more than 4 keypoints are detected
                if len(detected_labels) > 3:
                    h, mask, detected_labels_prev, detected_labels_src_pts_prev = ct.calculate_homography(frame_nbr, detected_labels, detected_labels_src_pts, detected_labels_dst_pts,
                         detected_labels_prev, detected_labels_src_pts_prev, keypoints_displacement_mean_tol)

                if  h.any():
                    ## Extract detections information for players
                    results_players, bboxes_p, labels_p, confs_p, detected_ppos_src_pts, detected_ball_src_pos = dt.detect_players(
                        frame, model_players, player_model_conf_thresh)

                    if detected_ball_src_pos is None:
                        nbr_frames_no_ball += 1
                    else:
                        nbr_frames_no_ball = 0

                    # Transform players and ball coordinates from frame plane to tactical map plan using the calculated Homography matrix
                    pred_dst_pts, detected_ball_dst_pos, ball_track_history = ct.transform_coordinates(h, detected_ppos_src_pts, detected_ball_src_pos, ball_track_history, ball_track_dist_thresh, max_track_length, show_b)

                # Players Team Prediction
                players_teams_list, annotated_frame, obj_palette_list = tp.predict_team(frame, labels_p, bboxes_p, color_list_lab, nbr_team_colors)

                # Updated Frame with annotations
                annotated_frame, tac_map_copy = at.annotate_frame(annotated_frame, bboxes_p, confs_p, labels_p, obj_palette_list, colors_dic, players_teams_list,
                   detected_labels_src_pts, pred_dst_pts, labels_dic, detected_ball_src_pos, detected_ball_dst_pos, bboxes_k, tac_map_copy, show_pal, show_k, show_p, h)

                # Updated Tactical Map with annotations
                prev_frame_time, final_img = at.annotate_tactical_map(tac_map_copy, annotated_frame, ball_track_history, prev_frame_time)

                # Display of the annotated frame
                stframe.image(final_img, channels="BGR")

                if save_output:
                    output.write(cv2.resize(final_img, (width, height)))

        # Remove progress bar and return
        st_prog_bar.empty()

        status = True
    else:
        try:
            # Release the video capture object and close the display window
            cap.release()
        except:
            pass
    if status:
        st.toast(f'Detection Completed !')
        cap.release()


if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass