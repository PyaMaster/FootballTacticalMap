# team_prediction.py

# Import libraries
import numpy as np
import pandas as pd
import cv2
import skimage
from PIL import Image, ImageColor

nbr_team_colors = 2

def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

    colors_dic = {
        team1_name: [team1_p_color_rgb, team1_gk_color_rgb],
        team2_name: [team2_p_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name] + colors_dic[
        team2_name]  # Define color list to be used for detected player team prediction
    color_list_lab = [skimage.color.rgb2lab([i / 255 for i in c]) for c in
                      colors_list]  # Converting color_list to L*a*b* space
    return colors_dic, color_list_lab

def predict_team(frame, labels_p, bboxes_p, color_list_lab):
    # Implement team prediction logic here

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    obj_palette_list = []  # Initialize players color palette list
    palette_interval = (0, 5)  # Color interval to extract from dominant colors palette (1rd to 5th color)
    annotated_frame = frame

    ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
    for i, j in enumerate(labels_p):
        if int(j) == 0:
            bbox = bboxes_p[i, :]  # Get bbox info (x,y,x,y)
            obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]  # Crop bbox out of the frame
            obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
            center_filter_x1 = np.max([(obj_img_w // 2) - (obj_img_w // 5), 1])
            center_filter_x2 = (obj_img_w // 2) + (obj_img_w // 5)
            center_filter_y1 = np.max([(obj_img_h // 3) - (obj_img_h // 5), 1])
            center_filter_y2 = (obj_img_h // 3) + (obj_img_h // 5)
            center_filter = obj_img[center_filter_y1:center_filter_y2,
                            center_filter_x1:center_filter_x2]
            obj_pil_img = Image.fromarray(np.uint8(center_filter))  # Convert to pillow image

            reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)  # Convert to web palette (216 colors)
            palette = reduced.getpalette()  # Get palette as [r,g,b,r,g,b,...]
            palette = [palette[3 * n:3 * n + 3] for n in range(256)]  # Group 3 by 3 = [[r,g,b],[r,g,b],...]
            color_count = [(n, palette[m]) for n, m in
                           reduced.getcolors()]  # Create list of palette colors with their frequency
            RGB_df = pd.DataFrame(color_count, columns=['cnt', 'RGB']).sort_values(
                # Create dataframe based on defined palette interval
                by='cnt', ascending=False).iloc[
                     palette_interval[0]:palette_interval[1], :]
            palette = list(RGB_df.RGB)  # Convert palette to list (for faster processing)
            annotated_frame = cv2.rectangle(annotated_frame,  # Add center filter bbox annotations
                                            (int(bbox[0]) + center_filter_x1,
                                             int(bbox[1]) + center_filter_y1),
                                            (int(bbox[0]) + center_filter_x2,
                                             int(bbox[1]) + center_filter_y2), (0, 0, 0), 2)

            # Update detected players color palette list
            obj_palette_list.append(palette)

        ## Calculate distances between each color from every detected player color palette and the predefined teams colors
        players_distance_features = []
        # Loop over detected players extracted color palettes
        for palette in obj_palette_list:
            palette_distance = []
            palette_lab = [skimage.color.rgb2lab([i / 255 for i in color]) for color in
                           palette]  # Convert colors to L*a*b* space
            # Loop over colors in palette
            for color in palette_lab:
                distance_list = []
                # Loop over predefined list of teams colors
                for c in color_list_lab:
                    # distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                    distance = skimage.color.deltaE_cie76(color,
                                                          c)  # Calculate Euclidean distance in Lab color space
                    distance_list.append(distance)  # Update distance list for current color
                palette_distance.append(distance_list)  # Update distance list for current palette
            players_distance_features.append(palette_distance)  # Update distance features list

        ## Predict detected players teams based on distance features
        players_teams_list = []
        # Loop over players distance features
        for distance_feats in players_distance_features:
            vote_list = []
            # Loop over distances for each color
            for dist_list in distance_feats:
                team_idx = dist_list.index(
                    min(dist_list)) // nbr_team_colors  # Assign team index for current color based on min distance
                vote_list.append(team_idx)  # Update vote voting list with current color team prediction
            players_teams_list.append(
                max(vote_list, key=vote_list.count))  # Predict current player team by vote counting

        return players_teams_list, annotated_frame, obj_palette_list
