import argparse
import os
from shutil import copy
def refine_dataset(args):
    root_dir = args.dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    deep = args.deep
    folders = [""]
    for i in range(deep):
        new_folders = []
        for folder in folders:
            names = sorted(os.listdir(os.path.join(root_dir,folder)))
            for name in names:
                new_path =os.path.join(save_dir,folder,name)
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                new_folders.append(os.path.join(folder,name))
        folders = new_folders
    for folder in folders:
        for i,frame in enumerate(sorted(os.listdir(os.path.join(root_dir,folder)))):
            copy(os.path.join(root_dir,folder,frame),os.path.join(save_dir,folder,str(i).zfill(6)+"."+frame.split(".")[-1]))
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Path to dataset.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to dataset.')
    parser.add_argument('--deep', type=int, default=1, help='The depth of the file path from the root directory "dir" to the directory of a video.')
    args = parser.parse_args()
    refine_dataset(args)
    