import cv2  
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from keras import preprocessing
from firebase import firebase

class_names = ['S1', 'S2', 'S3']
camera = cv2.VideoCapture(0)
camera_height = 500
raw_frames_type_1 = []
firebase = firebase.FirebaseApplication('https://android-studio-351c1.firebaseio.com/')

def InsertOrUpdate(Val):
    result = firebase.post('/Hack', {'Cond': Val})
    print("working: ",result)

while(True):
    # read a new frame
    _, frame = camera.read()
    
    # flip the frame
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))

    # add rectangle
    cv2.rectangle(frame, (150, 75), (650, 425), (0, 255, 0), 2)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    HSV_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    HSL_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # show the frame
    cv2.imshow("RGB", frame)
    cv2.imshow("Gray", gray_image)
    cv2.imshow("HSV", HSV_image)
    cv2.imshow("Filter", HSL_image)

    key = cv2.waitKey(1)

    # quit camera if 'q' key is pressed
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("1"):
        # save the frame
        raw_frames_type_1.append(frame)
        print('1 key pressed - saved TYPE_1 frame')
        roi = frame[75+2:425-2, 300+2:650-2]
        
        # parse BRG to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # resize to 224*224
        roi = cv2.resize(roi, (80, 80))
        
        # persist the image
        img_used = 'images.jpg'
        cv2.imwrite(img_used.format(frame), cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        model = tf.keras.models.load_model("S2.h5", compile=False)
        type_1 = preprocessing.image.load_img(img_used, target_size=(80, 80))

        type_1_X = np.expand_dims(type_1, axis=0)
        predictions = model.predict(type_1_X)

        print('The specimen predicted is: {}'.format(class_names[np.argmax(predictions)]))
        if (format(class_names[np.argmax(predictions)]) == 'S1'):
            InsertOrUpdate(1)
        elif (format(class_names[np.argmax(predictions)]) == 'S2'):
            InsertOrUpdate(2)
        elif (format(class_names[np.argmax(predictions)]) == 'S3'):
            InsertOrUpdate(3)

camera.release()
cv2.destroyAllWindows()