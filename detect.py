import os
from pathlib import Path
import argparse
from detect_lib.model import YOLO_Detector
import time

def find_all_videos(root:os.PathLike, video_type:list)->list[Path]:
    
    ret = []
    for dirs, _, files in os.walk(root, topdown=True):
        if len(files) != 0:
            for posfix in video_type:
                ret += [_ for _ in Path(dirs).glob(f"*.{posfix}")]
    
    return ret

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=Path)
    parser.add_argument("--batch_size", type=int, default = 180)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_arguments()

    all_videos_path = find_all_videos(root = args.video_root, video_type=["mp4"])
    
    detector = YOLO_Detector()

    for video_path in all_videos_path:
        
        s = time.time()
        video_dir = video_path.parent
        video_name = video_path.stem
        frame_save_dir = video_dir/video_name
        frame_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"read video : {video_path} -> {frame_save_dir}")
        detector(
            video_path, 
            dst_dir = frame_save_dir, 
            batch_size = args.batch_size,
            wanted_fps=args.fps
        )
        e = time.time()
        print(f"consuming : {e-s:.4f} sec")