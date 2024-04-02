import os
import cv2
import numpy as np
import math
import threading 

def mutithread_imwrite(imgs:list[np.ndarray], paths:list[os.PathLike], MAX_Thread_num:int=10):
    
    write_times = math.ceil(len(paths)/MAX_Thread_num)
    for i in range(write_times):
        T = list(
            threading.Thread(target=cv2.imwrite, args=(paths[j], imgs[j])) 
            for j in range(MAX_Thread_num*i, min(MAX_Thread_num*(i+1), len(paths)))
        )
    
        for tid in range(len(T)):
            T[tid].start()
        
        for tid in range(len(T)):
            T[tid].join()


