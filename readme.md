# Yolov5 to LabelImg notation

Using ```yolov5x``` to detect videos and then write all the frames and detected results in CreateML notation form.

- It is for LabelImg : https://github.com/tzutalin/labelImg

## Environment:
- python3
    - torch
    - ultralytics
    - opencv-python
    - tqdm  

** Recommand running on __Linux__ 

## Execution:

Arguments:
- --video_root :
    - The root of all the videos you want to detect
- --batch_size:
    - the batch size for yolov5x, default is 180
- device :
    - e.g. cpu, cuda:0, cuda:1, ..., default is cuda:0

E.g.
```python detect.py --video_root ./all_videos/ --batch_size 180 --device cuda:0```

It will generate a folder named according to video name for each video under ```./all_videos/```

E.g. 
video_root : ./all_videos/ :

```
.
├── all_videos/
│   ├── 1/
│   │   └── c1.mp4
│   ├── 2/
│   │   └── c2.mp4
│   └── 3/
│       └── c3.mp4  
└── ...
```

Then it will generate the following folder under this root:

```
.
├── all_videos/
│   ├── 1/
│   │   ├── c1.mp4
│   │   └── c1/
│   │       ├── c1_frame1.jpg
│   │       ├── c1_frame1.json
│   │       ├── c1_frame2.jpg
│   │       ├── c1_frame2.json
│   │       └── ...
│   ├── 2/
│   │   ├── c2.mp4
│   │   └── c2/
│   │       ├── c2_frame1.jpg
│   │       ├── c2_frame1.json
│   │       ├── c2_frame2.jpg
│   │       ├── c2_frame2.json
│   │       └── ...
│   └── 3/
│       ├── c3.mp4  
│       └── c3/
│           ├── c3_frame1.jpg
│           ├── c3_frame1.json
│           ├── c2_frame2.jpg
│           ├── c2_frame2.json
│           └── ...
└── ...
```