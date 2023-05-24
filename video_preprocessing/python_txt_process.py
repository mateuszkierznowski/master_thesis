import os
import pathlib

path_val: str = r'C:\Users\User\Desktop\Master_videos_all\val_2'
path_train: str = r'C:\Users\User\Desktop\Master_videos_all\train_2'
path_final: str = r"C:\Users\User\Desktop\altered_skeleton"
path_final_2: str = r"C:\Users\User\Desktop\human"

list_folders: list = os.listdir(path_val)

#path_vid = path_train
folder_name = 'train'

for folder in list_folders:
    pathlib.Path(f"{path_final}/train/{folder}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{path_final}/test/{folder}").mkdir(parents=True, exist_ok=True)


for folder in list_folders:
    pathlib.Path(f"{path_final_2}/train/{folder}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{path_final_2}/test/{folder}").mkdir(parents=True, exist_ok=True)

def create_ps_script(script_name: str, path_vid: str, path_final_, type_: str, train_or_valid: str) -> None:
    script = open(script_name, 'w')
    print(path_vid)
    for folder in list_folders:
        #print(f"{folder} folder")
        videos_within_folder: list = os.listdir(f"{path_vid}/{folder}")
        for video in videos_within_folder:
            script.write(f"python create_ps1_file.py --type {type_} --input {path_vid}/{folder}/{video} --output {path_final_}/{train_or_valid}/{folder}/{video} \n")

    script.close()


create_ps_script("video_preprocessing/script_ps_skeleton_train.ps1", r'C:\Users\User\Desktop\Master_videos_all\train_2',\
                 r"C:/Users/User/Desktop/altered_skeleton", "skeleton", "train")

create_ps_script("video_preprocessing/script_ps_skeleton_test.ps1", r'C:\Users\User\Desktop\Master_videos_all\val_2', \
                 r"C:/Users/User/Desktop/altered_skeleton", "skeleton", "test")

create_ps_script("video_preprocessing/script_ps_human_trainps1", r'C:\Users\User\Desktop\Master_videos_all\train_2', \
                 r"C:/Users/User\Desktop\human", "human", "train")

create_ps_script("video_preprocessing/script_ps_human_test.ps1", r'C:\Users\User\Desktop\Master_videos_all\val_2', \
                 r"C:\Users\User\Desktop\human", "human", "test")