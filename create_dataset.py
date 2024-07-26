import os
import pickle
import numpy as np

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# mediapipe hands module 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# declaring the mediapipe model 
# since serving static images, set static_image_mode to True
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# for all the subdirectories (i.e. classes) in data directory
for dir_ in os.listdir(DATA_DIR):
    # for all images in each class 
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): 

        # initialise empty normalised keypoints array
        data_norm = []

        # initialise empty x and y array 
        x_ = [] 
        y_ = []

        # get the image invidually by looping 
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # convert image to rng 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # process the image 
        results = hands.process(img_rgb)

        # if results exists then...
        if results.multi_hand_landmarks:

            # initialise empty keypoints array  
            

            # FOR ALL HAND LANDMARKS IN THE RESULTS
            # image_height, image_width, _ = img_rgb.shape
            # annotated_image = img_rgb.copy()
            # for hand_landmarks in results.multi_hand_landmarks:
            #     print('hand_landmarks:', hand_landmarks)
            #     print(
            #         f'Index finger tip coordinates: (',
            #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            #     )
            #     mp_drawing.draw_landmarks(
            #         annotated_image,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks: 
                for lm in hand_landmarks.landmark: 
                    
                    x = lm.x 
                    y = lm.y 

                    # append x and y values 
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)): 
                    x = hand_landmarks.landmark[i].x 
                    y = hand_landmarks.landmark[i].y 
                    data_norm.append(x - min(x_))
                    data_norm.append(y - min(y_))
            
            # append the normalised keypoints to the data array
            data.append(data_norm)
            labels.append(dir_)
            
        else: 
             keypoints = np.zeros(21*3)


# crate a pickle file to store the data
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
# close the file 
f.close()

        
    
