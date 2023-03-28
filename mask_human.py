import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    enable_segmentation=True,
    min_detection_confidence=.3,
    min_tracking_confidence=.1)

cap = cv2.VideoCapture(
    r'C:\Users\User\Desktop\Master_videos_all\train\Squad\317115823_1310457289755916_8889208919929233640_n.mp4')
writer = cv2.VideoWriter("Filenames/person_masks.avi", cv2.VideoWriter_fourcc(*'XVID'), 24, (600, 800))
pose_landmark_lst = []
frame_lst = []
results_lst = []
# create capture object
while cap.isOpened():
    # read frame from capture object
    _, frame = cap.read()
    print(_)
    if _:
        frame_lst.append(frame)
        # convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = pose.process(RGB)
        results_lst.append(results)
        if results.segmentation_mask is not None:
            rgb = cv2.cvtColor(results.segmentation_mask, cv2.COLOR_GRAY2RGB)
            rgb = rgb * 255
            rgb = rgb.astype(np.uint8)
            #fin_ = np.where(rgb != 0, frame_lst[frame_no], rgb)
            frame = np.where(rgb != 0, frame, 255)
        else:
            black = np.zeros(frame.shape, dtype=np.uint8)

        # show the final output
        cv2.imshow('Output', frame)


        frame = cv2.resize(frame, (600, 800))
        writer.write(frame)


        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# writer.release()
cap.release()
cv2.destroyAllWindows()

frame_no = 270
rgb = cv2.cvtColor(results_lst[frame_no].segmentation_mask, cv2.COLOR_GRAY2RGB)
plt.imshow(rgb)
rgb = rgb * 255
rgb = rgb.astype(np.uint8)
#fin_ = np.where(rgb != 0, frame_lst[frame_no], rgb)
fin_ = np.where(rgb != 0, frame_lst[frame_no], 255)

plt.imshow(fin_[:, :, ::-1])
plt.savefig("Filenames/squad_all_mask.png")
plt.show()

plt.imshow(frame_lst[270][:, :, ::-1])
plt.savefig("Filenames/squad_whole.png")
plt.show()

cv2.imshow("Mask squad", fin_)
cv2.W
# pose_landmark
sth = pose_landmark_lst[0].landmark
