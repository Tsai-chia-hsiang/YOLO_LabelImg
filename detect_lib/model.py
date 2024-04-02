import os
import json
import cv2
import torch
import numpy as np
from tqdm import trange
import os.path as osp
from .image_utils import mutithread_imwrite

class YOLO_Detector():
    
    def __init__(self, wanted_classes:list = None, device:str="cpu") -> None:
    
        self.model = self._get_yolov5_model(to_device=device)
        self.wanted_classes = wanted_classes if wanted_classes is not None else [0,1,2,3,5,7]

    def _get_yolov5_model(self, to_device):
        
        model = torch.hub.load(
            repo_or_dir='ultralytics/yolov5', 
            model='yolov5x', pretrained=True
        )
        model.to(device = torch.device(to_device))
        
        return model
    
    def __call__(self, src:os.PathLike, dst_dir:os.PathLike, **kwargs) -> None:
        
        if src[-4:] == ".mp4":
            self._read_video_frames_and_detection(
                video_path = src, dst_dir = dst_dir, 
                batch_size = kwargs["batch_size"]
            )
        else:
            raise(ValueError("Not support this type"))
    
    def _a_batch(self,frame_names:list[os.PathLike], frame_buf:list[np.ndarray])->None:
        
        mutithread_imwrite(
            imgs = frame_buf, 
            paths = frame_names, 
            MAX_Thread_num = os.cpu_count()//2 
        )
        for idx, ann in enumerate(self.detect_bbox(frame_buf, frame_names)):
            fname = frame_names[idx]
            #cv2.imwrite(fname, frame_buf[idx])
            with open(f"{fname[:fname.rfind('.')]}.json", "w+") as jf:
                json.dump(ann, jf, indent=4, ensure_ascii=False)
    
    def _read_video_frames_and_detection(self, video_path:os.PathLike, dst_dir:os.PathLike, batch_size:int, **kwargs)->None:
        """
        This member function should be called by 
        __call__() function, do not call it manually

        read all frame from src video by cv2.VideoCapture()
        and detection objects from them then write all frames with their annotation 
        to dst folder
        """
        def set_sampling_freq(video_fps:int)->int:
            if "wanted_fps" in kwargs:
                return video_fps // kwargs["wanted_fps"] \
                    if kwargs["wanted_fps"] <= video_fps else 1
            else:
                return 1
        
        video_name =  osp.split(video_path)[-1]
        video_name = video_name[:video_name.rfind(".")]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video {video_path}")
            return
        
        freq:int = set_sampling_freq(video_fps = int(cap.get(cv2.CAP_PROP_FPS))) 
        total_frames:int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        frame_buf, frame_names = [None]*batch_size, [""]*batch_size
        pbar = trange(total_frames)
        
        torch.enable_grad(False)

        for i in pbar:

            pbar.set_postfix(ordered_dict={"phase":"read frames"})
            _, frame = cap.read()

            if i % freq == 0:
                pbar.set_postfix(ordered_dict={"phase":"buffing"})
                frame_buf[i%batch_size] = frame
                frame_names[i%batch_size] = osp.join(
                    dst_dir, f"{video_name}_{frame_idx:08d}.jpg"
                )
                frame_idx += 1
            
            if (i+1) % batch_size == 0:
                pbar.set_postfix(ordered_dict={"phase":"detection"})
                self._a_batch(frame_buf = frame_buf, frame_names = frame_names)
        
        
        remaining = (i+1) % batch_size

        if remaining:
            self._a_batch(
                frame_buf = frame_buf[:remaining], 
                frame_names = frame_names[:remaining]
            )
        
        cap.release()   
        
    def detect_bbox(self, imgs:list[np.ndarray], img_names:list[str])->list:

        def parse_xyxy(xyxy:torch.Tensor)->list[dict]:
            """
            one image
            xyxy : xmin, ymin, xmax, ymax, confidence, class_id 
            """
            return list(
                {   
                    "label":self.model.names[int(j[5])],
                    "coordinates":{
                        "x":float((j[0] + j[2])/2),
                        "y":float((j[1] + j[3])/2),
                        "width":float(j[2] - j[0]),
                        "height":float(j[3] - j[1])
                    }
                } for j in xyxy.cpu().numpy() \
                    if int(j[5]) in self.wanted_classes
            )
            

        results = self.model(imgs)
        create_ML_notations = [None]*len(imgs)
 
        for i, xyxy in enumerate(results.xyxy):
            create_ML_notations[i] = [{
                "image": osp.split(img_names[i])[-1], 
                "annotations":parse_xyxy(xyxy = xyxy)
            }]

        return create_ML_notations 

