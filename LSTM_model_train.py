import pickle 
import cv2 
import mediapipe as mp 
import numpy as np

model_dict = pickle.load(open('neuralmodel.pickle', 'rb'))
model = model_dict['model']

vid = cv2.VideoCapture(0)

# mediapipe hands module 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# declaring the mediapipe model 
# since serving static images, set static_image_mode to True
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}
while True: 

    # we did this previous to train the model with data, we are doing this again to predict the results with new data
    # initiallise empty normalised keypoints array
    data_aux = []

    # creating empty arrays for x and y keypoints coordinates 
    x_ = []
    y_ = []

    # capturing the video frame by frame
    ret, frame = vid.read()

    # getting the width and height of frame 
    H, W, _ = frame.shape 

    # convert frame image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame 
    results = hands.process(frame_rgb)

    # if results exists then...
    if results.multi_hand_landmarks: 

        # first loop to draw the hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks: 
            
            # drawing the hand landmarks on the frame 
            # refer create-dataset.py commented section 
            mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # second loop to get the x and y coordinates of the hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark: 
                    
                    x = lm.x 
                    y = lm.y 

                    # append x and y values 
                    x_.append(x)
                    y_.append(y)

            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure data_aux has 42 values
        if len(data_aux) == 42:
            data_aux = np.array(data_aux).reshape(1, 1, 42)  # Reshape to (1, 1, 42)

            # predict the results from the model and covert data_aux to numpy array
            prediction = model.predict(data_aux)

            # since these are probabilities, get the index of the highest probability to convert to labels.
            predicted_index = np.argmax(prediction, axis=1)[0]

            # get the label from the label dictionary given the predicted index.
            predicted_character = labels_dict[int(predicted_index)]

        
        # get the size of predicting rectangle 
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # draw the rectangle 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # write the predicted character on the rectangle frame
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
            
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv2.destroyAllWindows()