import os
import sys
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    enable_segmentation=True,
    min_detection_confidence=.3,
    min_tracking_confidence=.1)

cap = cv2.VideoCapture(r'C:\Users\User\Desktop\Master_videos_all\train\Squad\317115823_1310457289755916_8889208919929233640_n.mp4')
writer = cv2.VideoWriter("Filenames/example_with_skeleton_only_fv.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (600, 800))
pose_landmark_lst = []

# create capture object
while cap.isOpened():
    # read frame from capture object
    _, frame = cap.read()
    print(_)
    if _:
        # convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = pose.process(RGB)
        pose_landmark_lst.append(results.pose_landmarks)
        #print(results.pose_landmarks)
        # draw detected skeleton on the frame
        #frame.fill(255)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # show the final output
        cv2.imshow('Output', frame)
        frame = cv2.resize(frame, (600, 800))
        #writer.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

#writer.release()
cap.release()
cv2.destroyAllWindows()


#pose_landmark
sth = pose_landmark_lst[0].landmark
