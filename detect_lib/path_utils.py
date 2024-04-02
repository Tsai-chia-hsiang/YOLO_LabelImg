import os
from glob import glob
import os.path as osp


def find_all_videos(root:os.PathLike, video_type:list)->list[os.PathLike]:
    ret = []
    for dirs, _, files in os.walk(root, topdown=True):
        if len(files) != 0:
            for f in video_type:
                ret += glob(osp.join(dirs, f'*.{f}'))
    return ret

def create_path(p:os.PathLike)->os.PathLike:
    if not os.path.exists(p):
        os.makedirs(p)
    return p