# AI-Powered Football Tactical Map
## Overview
FootballTacticalMap is a Streamlit-based web application that leverages computer vision and deep learning to analyze football tactics from tactical camera footage. 
It detects players, predicts team affiliations, and overlays tactical insights onto a football pitch map. The application provides valuable insights to assist in decision-making processes for coaches, analysts, and enthusiasts.

## 📌 Features
- **Object Detection:** Identify players, referees, and the ball within match footage.
- **Team Classification:** Assign players to their respective teams based on predefined team colors.
- **Tactical Mapping:** Generate a tactical map representation of player positions and movements.
- **Ball Tracking:** Monitor and visualize the ball's trajectory throughout the match.

## 🗂 Project Structure
```
📂 FootballTacticalMap/
├── app.py                           # Main Streamlit app
├── 📂 annotation/                     # Annotation functions and drawing utilities
│   └── 📄 annotation.py
├── 📂 assets/                         # Static assets (images, test video)
│   ├── 📄 tactical_map.jpg
│   └── 📄 test_vid.mp4
├── 📂 config/                         # YAML config files and label mappings
│   ├── 📄 config pitch dataset.yaml
│   ├── 📄 config players dataset.yaml
│   ├── 📄 environment.yml
│   └── 📄 pitch map labels position.json
├── 📂 coordinate_transformer/
│   └── 📄 coordinate_transformer.py   # Converts image coordinates to pitch coordinates
├── 📂 Detection/
│   └── 📄 detection.py                # YOLO-based object detection
├── 📂 models/                         # Pre-trained model weights
│   ├── 📄 Field_Keypoints.pt
│   └── 📄 Players.pt
├── 📂 outputs/                        # Generated output videos
│   ├── 📄 Demo_result.mp4
│   ├── 📄 detect_1.mp4
│   └── 📄 detect_2.mp4
├── 📂 team_prediction/
│   └── 📄 team_prediction.py         # Player team prediction logic
└── 📂 .venv/                          # Python virtual environment
```

## 🛠️ Installation
1. Clone the repo
```bash
git clone https://github.com/PyaMaster/FootballTacticalMap.git
cd FootballTacticalMap
```
