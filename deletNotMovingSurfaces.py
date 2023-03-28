import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(r'C:\Users\User\Desktop\Master_videos_all\train\Squad\317115823_1310457289755916_8889208919929233640_n.mp4')
#writer = cv2.VideoWriter("Filenames/example_with_skeleton_only_fv.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (600, 800))

frames_lst = []
frames_all = []
pose_landmark_lst = []
difference_frames = 10
difference_colors = 100

# create capture object
while cap.isOpened():
    # read frame from capture object
    _, frame = cap.read()
    if _:
        if len(frames_lst) > difference_frames:
            # convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_lst.append(frame)

            fr_1 = frames_lst[0]
            fr_2 = frames_lst[difference_frames]

            frames_indexes = fr_2 - fr_1
            frames_indexes = np.ceil(frames_indexes)

            fin_mat = np.where(frames_indexes > difference_colors, fr_2, 0)


            #frames_all.append(gray)

            cv2.imshow('Output', fin_mat)

            frames_lst = frames_lst[1:]
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_lst.append(frame)
    else:
        break
cap.release()
cv2.destroyAllWindows()


#second try


from PIL import Image
from transparent_background import Remover

cap = cv2.VideoCapture(r'C:\Users\User\Desktop\Master_videos_all\train\BarbellCurl\Snapinsta.app_20249544_453737998315467_2525946790209912832_n_Trim4.mp4')  # video reader for input
fps = cap.get(cv2.CAP_PROP_FPS)
remover = Remover()
writer = None
writer = cv2.VideoWriter("Filenames/example_with_skeleton_only_fv.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (600, 800))
i = 0
while cap.isOpened():
    ret, frame = cap.read()  # read video

    if ret is False:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame).convert('RGB')

    # if writer is None:
    #     writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
    #                              img.size)  # video writer for output

    out = remover.process(img, type='map')  # same as image, except for 'rgba' which is not for video.
    writer.write(cv2.cvtColor(cv2.resize(out, (600, 800)), cv2.COLOR_BGR2RGB))
    cv2.imshow('Output', out)
    i += 1
    print(i)

cap.release()
writer.release()