import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pathlib

path_val: str = r'C:\Users\User\Desktop\Master_videos_all\val'
path_train: str = r'C:\Users\User\Desktop\Master_videos_all\train'
path_final: str = r"C:\Users\User\Desktop\altered_dataset_with sekeleton"


def transfer_video(path_to_video: str, path_final: str):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        enable_segmentation=True,
        min_detection_confidence=.3,
        min_tracking_confidence=.1)

    cap = cv2.VideoCapture(path_to_video)

    mp4_to_avi: str = path_final.replace("mp4", 'avi')
    writer = cv2.VideoWriter(mp4_to_avi, cv2.VideoWriter_fourcc(*'XVID'), 30, (600, 600))
    pose_landmark_lst = []

    # create capture object

    while cap.isOpened():
        # read frame from capture object
        _, frame = cap.read()
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
            writer.write(frame)

            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

def main(folder_name: str, list_folders: list):

    for folder in list_folders:
        print(f"{folder} folder")
        folder_with_videos: str = f"{list_folders}/{folder}"
        videos_within_folder: list = os.listdir(f"{path_val}/{folder}")

        for video in videos_within_folder:
            print(video)
            try:
                pathlib.Path(f"{path_final}/{folder_name}/{folder}").mkdir(parents=True, exist_ok=True)
                transfer_video(f"{path_val}/{folder}/{video}", f"{path_final}/{folder_name}/{folder}/{video}")
            except RuntimeError:
                continue


lst_val_folders: list = os.listdir(path_val)
main('val', lst_val_folders)

lst_train_folders: list = os.listdir(path_train)
main('train', lst_train_folders)


#pose_landmark
#sth = pose_landmark_lst[0].landmark
