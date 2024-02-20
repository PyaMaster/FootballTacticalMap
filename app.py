import tempfile
import numpy as np

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
import Detection.detection as dt
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

    st.sidebar.markdown('---')
    st.sidebar.subheader("Team Names")
    team1_name = st.sidebar.text_input(label='First Team Name', value=demo_team_info["team1_name"])
    team2_name = st.sidebar.text_input(label='Second Team Name', value=demo_team_info["team2_name"])
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
                    **There is a demo videos that are automaticaly loaded when you start the app, alongside the recommended settings and hyperparameters**
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


if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass