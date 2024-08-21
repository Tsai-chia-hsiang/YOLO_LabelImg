# Ultralytics Yolov10l to LabelImg notation

Using ```yolov10l.pt``` to detect the vehicle-like objects from each frame of the video clips and write all the frames and detected results in __CreateML__ format.

- It is for [LabelImg](https://github.com/HumanSignal/labelImg.git)

## Environment:

** Recommand running on __Linux__ 

## Execution:

- Arguments:
    - --video_root :
        - The root of all the videos you want to detect. 
        - it refers to a day in our case.
            - it contains all the footage of the cameras collected from that day.
            - Each camera footage is in a folder that named after the camera ID.
    - --batch_size:
        - the number of images for detecting onces, default is 180

- About device:

    - Due to the implementation of Ultralytics, please use ```CUDA_VISIBLE_DEVICES=want_device_id``` to control if don't want use default device (cuda 0) 

E.g.
```
CUDA_VISIBLE_DEVICES=3 python detect.py --video_root /root/to/a/day/ --batch_size 180 
```

It will generate a folder named according to video name for each video under ```/root/to/a/day/```

E.g. 
video_root : ./dataset/0925/ :
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

Then it will generate the following folders under this root:
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

```
pip3 install labelImg
labelImg
```

- __The staff member's ID must remain consistent throughout their entire corresponding sequences.__

## The format conversion can be executed ONLY AFTER all staff have finished labeling.

# Format conversion

- The staff track IDs must remain the same throughout the day.
- since we only detect and track car ( single class ), the class ID is always 0 for each format

The program ```convert_tid.py``` can convert CreateML notation to :


## MOT2D format
```
python convert_tid.py --staff_annotation_root ./staff_bbox_annotation/
--to_root ./serial_number_trackid/ __--format mot2d__ 
```
Please note that frame ID starts from 0 but not 1 for each sequence.

E.g. 
```
python convert_tid.py --staff_annotation_root ./dataset/0925/ --to_root ./dataset_mot2d/0925/ --format mot2d
```

Then it will generate the following folders under this root:
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

converted annotation for each object : 

`0 normalized_center_x normalized_center_y normalized_width normalized_height serial_num_track_id`

This conversion will copy the file structure from `--staff_annotation_root` to `--to_root`, and it ensures all annotation files retain the same prefix as their corresponding frames.

- Please note that we just adding serial_num_track_id to yolo detection labels format
    - If you want to use this to train YOLO detector, please remove the `serial_num_track_id` at the end of each notation by your own. 
- About the corresponding images, we don't copy from `--staff_annotation_root` to `--to_root`. Therefore, you will need to move or copy those frames yourself if you want to train the YOLO detector.


E.g. 
```
python convert_tid.py --staff_annotation_root ./dataset/1015/ --to_root ./dataset_yolotid/0925/  --imgsz 1920 1080
```

Then it will generate the following folders under this root:
```
.
├── dataset_yolotid/
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

