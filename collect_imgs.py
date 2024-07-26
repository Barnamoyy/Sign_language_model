import os 
import cv2 

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# basically how many hand signs we need 
number_of_classes = 26

# how many data samples for each class 
dataset_size = 100 

# open the camera 
vid = cv2.VideoCapture(0)
# loop through for all signs individually 
for i in range(11, number_of_classes): 
    # if the directory does not exist, create it
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))
    
    print(f'Collecting images for class {i}')

    # loop through for each sample
    done = False 
    while True: 
        # capture the video frame by frame 
        ret, frame = vid.read()

        # put text in the video 
        # cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA) 
        cv2.putText(frame, "Press 'Q' to start", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        # if Q was pressed break this video 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    # keep track of data for each sign using counter variable 
    counter = 0
    while counter < dataset_size: 
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # save the frame 
        cv2.imwrite(os.path.join(DATA_DIR, str(i), f'{counter}.jpg'), frame)

        # increment counter 
        counter += 1 

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
