import os
import sys
import cv2
import keras
import numpy as np

exercises_dct = {0: 'BarbellCurl',
                 1: 'BenchPress',
                 2: 'BendOverRows',
                 3: 'Deadlift',
                 4: 'HipThrust',
                 5: 'KettleSwing',
                 6: 'MilitaryPress',
                 7: 'Pullup',
                 8: 'Pushup',
                 9: 'Squad'}

model = keras.models.load_model(r"C:\Users\User\Desktop\Master_models\model_2")

#cap = cv2.VideoCapture(r'C:\Users\User\Desktop\Master_videos_all\train\BarbellCurl\Snapinsta.app_10000000_153148417226393_4262127008987914977_n_Trim6.mp4')
cap = cv2.VideoCapture(0)

frames_lst = []
frames_all = []
exercise = None
# create capture object
while cap.isOpened():
    # read frame from capture object
    _, frame = cap.read()
    if _:
        frame_ = frame
        frame = cv2.resize(frame, (224, 224))
        frames_lst.append(format_frames(frame, (224, 224)))

        if len(frames_lst) == 10:
            example = np.array(frames_lst)
            np.expand_dims(example, axis=0)
            fin = model.predict_on_batch(np.expand_dims(example, axis=0))
            exercise = exercises_dct[fin.argmax()]
            print(exercise)
            frames_lst = []

        #cv2.imshow('Output', frame_)
        # convert the frame to RGB format
        org = (20, 20)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        if exercise:
            frame_ = cv2.putText(frame_, exercise, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('Output', frame_)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# pose_landmark
import matplotlib.pyplot as plt

plt.imshow(frames_all[0])
plt.show()
model.predict(np.expand_dims(frames_all[10:20], axis=0))

arrays = [format_frames(frames_, (224, 224)) for frames_ in frames_all[0:10]]
result = np.array(arrays)[..., [2, 1, 0]]

model.predict(np.expand_dims(result, axis=0)).argmax()
