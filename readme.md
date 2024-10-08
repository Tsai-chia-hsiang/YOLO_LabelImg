# Ultralytics Yolov10l to LabelImg notation

Using ```yolov10l.pt``` to detect the vehicle-like objects from each frame of the video clips and write all the frames and detected results in __CreateML__ format.

- It is for [LabelImg](https://github.com/HumanSignal/labelImg.git)

## Environment:

** Recommand running on __Linux__ 
setup:

The suggestion python version is above 3.10

1. install [torch](https://pytorch.org/get-started/locally/)
2. install [ultralytics](https://docs.ultralytics.com/quickstart/)
3. additional packages:
    - tqdm : pip install tqdm 
    - opencv-python : pip install opencv-python




## Execution:

- Arguments:
    - --video_root :
        - The root of all the videos you want to detect. 
        - it refers to a day in our case.
            - It contains all the footage collected by the cameras from that day.
            - Each camera footage is in a folder named after the camera ID.
    - --batch_size:
        - the number of images for detecting once, default is 180
    - --fps:
        - wanted fps, can set lower than your source video to down sampling the frames.
- About device:

    - Due to the implementation of Ultralytics, please use ```CUDA_VISIBLE_DEVICES=want_device_id``` to control if you don't want to use the default device (cuda 0) 

E.g.
```
CUDA_VISIBLE_DEVICES=3 python detect.py --video_root /root/to/a/day/ --batch_size 180 --fps 10
```

It will generate a folder named according to the video name for each video under ```/root/to/a/day/```

E.g. 
video_root : ./dataset :
```
.
├──dataset/
│    ├── 0925/
│    │   ├── cam1/
│    │   │   └── c1.mp4
│    │   ├── cam2/
│    │   │   └── c2.mp4
│    │   └── cam3/
│    │       └── c3.mp4  
│    └── ...
└──...
```

Then, it will generate the following folders under this root:
```
.
├── dataset/
│    ├── 0925/
│    │   ├── cam1/
│    │   │   ├── c1.mp4
│    │   │   └── c1/
│    │   │       ├── c1_frame1.jpg
│    │   │       ├── c1_frame1.json
│    │   │       ├── c1_frame2.jpg
│    │   │       ├── c1_frame2.json
│    │   │       └── ...
│    │   ├── cam2/
│    │   │   ├── c2.mp4
│    │   │   └── c2/
│    │   │       ├── c2_frame1.jpg
│    │   │       ├── c2_frame1.json
│    │   │       ├── c2_frame2.jpg
│    │   │       ├── c2_frame2.json
│    │   │       └── ...
│    │   └── cam3/
│    │       ├── c3.mp4  
│    │       └── c3/
│    │           ├── c3_frame1.jpg
│    │           ├── c3_frame1.json
│    │           ├── c2_frame2.jpg
│    │           ├── c2_frame2.json
│    │           └── ...
│    └── ...
└── ...
```

## Please use LabelImg with the above CreateML annotation files.
[LabelImg](https://github.com/HumanSignal/labelImg.git)

Create a new environment for __python=3.8__
```
pip3 install labelImg
labelImg
```

__Please choose CreateML format before labeling__

<img src="./doc/labelimg.png"  width="800px"/>

### Opening a directory to label
<img src="./doc/labelimg_ins1.jpg">

<img src="./doc/labelimg_ins2.jpg" width="450px"/> <img src="./doc/labelimg_ins2_1.jpg" width="450px"/>

### ShortCut for LabelImg:
- press: __w__: draw a new bounding box
- press: __D__: next frame
- press: __A__: previous frame
- Ctrl+C : copy  bboxes
- Ctrl+V : past bboxes
- Ctrl+A : select all bboxes

** after using ctrl+A - ctrl+C - ctrl+V to copy all the bboxes from
the previous frame to current frame, pressing __D__ __A__ to refresh the 
bbox annotations.

- __The staff member's ID must remain consistent throughout their entire corresponding sequences.__

## The format conversion can be executed ONLY AFTER all staff have finished labeling.

# Format conversion

- The staff track IDs must remain the same throughout the day.
- since we only detect and track cars ( single class ), the class ID is always 0 for each format

The program ```convert_tid.py``` can convert CreateML notation to :


## MOT2D format
```
python convert_tid.py --staff_annotation_root ./staff_bbox_annotation/
--to_root ./serial_number_trackid/ __--format mot2d__ 
```
Please note that frame ID starts from 0 but not 1 for each sequence.

E.g. 
```
python convert_tid.py --staff_annotation_root ./dataset/ --to_root ./dataset_mot2d/ --format mot2d
```

Then, it will generate the following folders under this root:
```
.
├── dataset_mot2d/
│    ├── 0925/
│    │   ├── cam1/
│    │   │   ├──  c1/
│    │   │       ├── c1.txt
│    │   ├── cam2/
│    │   │   ├── c2/
│    │   │       ├── c2.txt
│    │   ├── cam3/
│    │   │   ├── c3/
│    │   │       ├── c3.txt
│    └── ...
└── ...
```

Each camera will generate a __MOT2D gt.txt__ format file.


## YOLO format
```
python convert_tid.py --staff_annotation_root ./staff_bbox_annotation/
--to_root ./serial_number_trackid/ --imgsz height width
```

Converted annotation for each object : 

`0 normalized_center_x normalized_center_y normalized_width normalized_height serial_num_track_id`

This conversion will copy the file structure from `--staff_annotation_root` to `--to_root`, and it ensures all annotation files retain the same prefix as their corresponding frames.

- Please note that we are just adding serial_num_track_id to yolo detection labels format
    - If you want to use this to train YOLO detector, please remove the `serial_num_track_id` at the end of each notation by your own. 
- About the corresponding images, we don't copy from `--staff_annotation_root` to `--to_root`. Therefore, you must move or copy those frames yourself if you want to train the YOLO detector.


E.g. 
```
python convert_tid.py --staff_annotation_root ./dataset --to_root ./dataset_yolo/  --imgsz 1920 1080
```

Then, it will generate the following folders under this root:
```
.
├── dataset_yolo/
│    ├── 0925/
│    │   ├── cam1/
│    │   │   ├──  c1/
│    │   │       ├── c1_frame1.txt
│    │   │       ├── c1_frame2.txt
│    │   │       └── ...
│    │   ├── cam2/
│    │   │   ├── c2/
│    │   │       ├── c2_frame1.txt
│    │   │       ├── c2_frame2.txt
│    │   │       └── ...
│    │   ├── cam3/
│    │   │   ├── c3/
│    │   │       ├── c3_frame1.txt
│    │   │       ├── c3_frame2.txt
│    │   │       └── ...
│    └── ...
└── ...
```

