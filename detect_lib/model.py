import os
import json
import cv2
import torch
import numpy as np
from tqdm import trange
from .image_utils import mutithread_imwrite
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from pathlib import Path
from typing import Optional

class YOLO_Detector():
    
    def __init__(self, wanted_classes:list = [2,5,7], conf:float=0.1) -> None:
        """
        Extend of YOLO from ultralytics
        """
        self.model = YOLO("yolov10l.pt")
        self.conf = conf
        self.wanted_classes = wanted_classes
    
    def _a_batch(self,frame_names:list[Path], frame_buf:list[np.ndarray])->None:
        
        mutithread_imwrite(
            imgs = frame_buf, 
            paths = frame_names, 
            MAX_Thread_num = os.cpu_count()//2 
        )
        for idx, ann in enumerate(self.detect_bbox(frame_buf, frame_names)):
            fname = frame_names[idx]
            #cv2.imwrite(fname, frame_buf[idx])
            with open(fname.with_suffix(".json"), "w+") as jf:
                json.dump(ann, jf, indent=4, ensure_ascii=False)
    
    def detect_bbox(self, imgs:list[np.ndarray], img_names:list[Path])->list:

        def parse_xyxy(box:Boxes)->list[dict]:
            
            return list(
                {   
                    "label":self.model.names[int(box.cls[i])],
                    "coordinates":{
                        "x":float(box.xywh[i][0]),
                        "y":float(box.xywh[i][1]),
                        "width":float(box.xywh[i][2]),
                        "height":float(box.xywh[i][3])
                    }
                } for i in range(box.xywh.shape[0])
            )
        
        results:list[Results] = self.model(
            source=imgs, conf=self.conf,
            classes=self.wanted_classes,
            verbose = False
        )
        create_ML_notations = [
            [
                {
                    "image": img_names[i].name, 
                    "annotations":parse_xyxy(box=ri.boxes.cpu().numpy())
                } 
            ]
            for i, ri in enumerate(results)
        ]

        return create_ML_notations 
    
    @torch.no_grad()
    def __call__(self, video_path:Path, dst_dir:Path, batch_size:int, wanted_fps:Optional[int]=None)->None:
        
        """
        read all frame from src video (.mp4 recommended) by cv2.VideoCapture()
        and detect objects from them.
        After that, write all frames with their annotation 
        to dst folder
        """
        
        def set_sampling_freq(video_fps:int)->int:
            if wanted_fps is not None:
                return video_fps // wanted_fps \
                    if wanted_fps <= video_fps else 1
            else:
                return 1
        
        video_name = video_path.stem
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video {video_path}")
            return
        #int(cap.get(cv2.CAP_PROP_FPS))
        freq = set_sampling_freq(video_fps = wanted_fps) 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        frame_buf, frame_names = [None]*batch_size, [None]*batch_size
        pbar = trange(total_frames)
        

        for i in pbar:

            pbar.set_postfix(ordered_dict={"phase":"read frames"})
            _, frame = cap.read()

            if i % freq == 0:
                pbar.set_postfix(ordered_dict={"phase":"buffering"})
                
                frame_buf[i%batch_size] = frame
                frame_names[i%batch_size] = dst_dir/f"{video_name}_{frame_idx:08d}.jpg"
                
                frame_idx += 1
            
            if (i+1) % batch_size == 0:
                pbar.set_postfix(ordered_dict={"phase":"detection"})
                self._a_batch(
                    frame_buf = frame_buf, 
                    frame_names = frame_names
                )
            
        remaining = (i+1) % batch_size

        if remaining:
            self._a_batch(
                frame_buf = frame_buf[:remaining], 
                frame_names = frame_names[:remaining]
            )
        
        cap.release()   
        