import os
from pathlib import Path

def find_all_file(root:os.PathLike, ftype:list)->list[Path]:
    
    ret = []
    for dirs, _, files in os.walk(root, topdown=True):
        if len(files) != 0:
            for posfix in ftype:
                ret += [_ for _ in Path(dirs).glob(f"*.{posfix}")]
    
    return ret