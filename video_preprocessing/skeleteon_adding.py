import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pathlib

path_val: str = r'C:\Users\User\Desktop\Master_videos_all\val_2'
path_train: str = r'C:\Users\User\Desktop\Master_videos_all\train_2'
path_final: str = r"C:\Users\User\Desktop\altered_skeleton"




def transfer_video(path_to_video: str, path_final: str, type: str = "skeleton"):
    print(f"currently processing: {path_to_video}")
    final_output: int = 224

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        enable_segmentation=True,
        min_detection_confidence=.2,
        min_tracking_confidence=.05)

    cap = cv2.VideoCapture(path_to_video)

    mp4_to_avi: str = path_final.replace("mp4", 'avi')
    writer = cv2.VideoWriter(mp4_to_avi, cv2.VideoWriter_fourcc(*'XVID'), 24, (final_output, final_output))
    pose_landmark_lst = []

    # create capture object
    while cap.isOpened():
        # read frame from capture object
        _, frame = cap.read()
        if _ and type == "skeleton":
            # convert the frame to RGB format
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process the RGB frame to get the result
            results = pose.process(RGB)
            #pose_landmark_lst.append(results.pose_landmarks)
            #print(results.pose_landmarks)
            # draw detected skeleton on the frame
            frame.fill(255)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            #cv2.imshow('Output', frame)
            frame = cv2.resize(frame, (final_output, final_output))
            writer.write(frame)
            # show the final output



        elif _ and type == "human":
            print(_)
            # convert the frame to RGB format
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # process the RGB frame to get the result
            results = pose.process(RGB)
            if results.segmentation_mask is not None:
                rgb = cv2.cvtColor(results.segmentation_mask, cv2.COLOR_GRAY2RGB)
                rgb = rgb * 255
                rgb = rgb.astype(np.uint8)
                # fin_ = np.where(rgb != 0, frame_lst[frame_no], rgb)
                frame = np.where(rgb != 0, frame, 255)
            else:
                black = np.zeros(frame.shape, dtype=np.uint8)

            if cv2.waitKey(1) == ord('q'):
                break

            #cv2.imshow('Output', frame)
            frame = cv2.resize(frame, (final_output, final_output))
            writer.write(frame)
        else:
            break


        # show the final output


    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    del pose

def main(folder_name: str, list_folders: list, type_: str, path_vid: str):

    for folder in list_folders:
        print(f"{folder} folder")
        videos_within_folder: list = os.listdir(f"{path_vid}/{folder}")

        for video in videos_within_folder:
            print(video)
            pathlib.Path(f"{path_final}/{folder_name}/{folder}").mkdir(parents=True, exist_ok=True)
            transfer_video(f"{path_vid}/{folder}/{video}", f"{path_final}/{folder_name}/{folder}/{video}", type_)


if __name__ == "__main__":
    type_ = 'skeleton'
    path_vid = path_train
    lst_val_folders: list = os.listdir(path_val)
    main('train', lst_val_folders, type_, path_vid)


    lst_train_folders: list = os.listdir(path_train)
    main('val', lst_train_folders, type, path_val)

    type_ = 'human'
    path_vid = "path/to/human_folder"

    lst_val_folders: list = os.listdir(path_val)
    main('train', lst_val_folders, type_, path_vid)

    lst_train_folders: list = os.listdir(path_train)
    main('train', lst_train_folders, type)

