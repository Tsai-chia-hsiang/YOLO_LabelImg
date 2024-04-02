import os.path as osp
import argparse
from detect_lib.path_utils import *
from detect_lib.model import YOLO_Detector


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str)
    parser.add_argument("--batch_size", type=int, default = 180)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_arguments()

    all_videos_path = find_all_videos(root = args.video_root, video_type=["mp4"])
    
    detector = YOLO_Detector(device=args.device)

    for video_path in all_videos_path:
        video_dir, video_name = osp.split(video_path)
        frame_save_dir = create_path(osp.join(video_dir, video_name[:video_name.rfind(".")]))
        print(f"read video :{video_path} -> {frame_save_dir}")
        detector(video_path, dst_dir = frame_save_dir, batch_size = args.batch_size)
