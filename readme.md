# Yolov5 to LabelImg notation

Using ```yolov5x``` to detect videos and then write all the frames and detected results in CreateML notation form.

- It is for [LabelImg](https://pypi.org/project/labelImg/)

## Due to implementation constraints, the staff track IDs must remain the same throughout the day.

## Environment:
- python3
- Requirments for the third-party packages : 
    - torch
    - ultralytics
    - opencv-python
    - tqdm  

** Recommand running on __Linux__ 

## Execution:

Arguments:
- --video_root :
    - The root of all the videos you want to detect. 
    - it refers to a day in our case.
        - it contains all the footage of the cameras collected from that day.
        - Each camera footage is in a folder that named after the camera ID.
- --batch_size:
    - the batch size for yolov5x, default is 180
- device :
    - e.g. cpu, cuda:0, cuda:1, ..., default is cuda:0

E.g.
```
python detect.py --video_root /root/to/a/day/ --batch_size 180 --device cuda:0
```

It will generate a folder named according to video name for each video under ```/root/to/a/day/```

E.g. 
video_root : ./dataset/1015/ :
```
.
├──dataset/
│    ├── 1015/
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
│    ├── 1015/
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

# Annotation convert

- Due to implementation constraints, the staff track IDs must remain the same throughout the day.
- since we only detect and track car ( single class ), the class ID is always 0 for each format

The program ```convert_tid.py``` can convert CreateML notation to :

## YOLO-tid format
python convert_tid.py --staff_annotation_root ./staff_bbox_annotation/
--to_root ./serial_number_trackid/ --imgsz height width

converted annotation for each object : 

`0 normalized_center_x normalized_center_y normalized_width normalized_height serial_num_track_id`

This conversion will copy the file structure from `--staff_annotation_root` to `--to_root`, and it ensures all annotation files retain the same prefix as their corresponding frames.

- Please note that we just adding serial_num_track_id to yolo detection labels format
    - If you want to use this for training YOLO, please remove the `serial_num_track_id` at the end using other method. 
- About the corresponding images, we don't copy from `--staff_annotation_root` to `--to_root`, so you need to move/copy those frames by your own if you want to train YOLO detector.

Sorry for the inconvenience.


E.g. 
```
python convert_tid.py --staff_annotation_root ./dataset/1015/ --to_root ./dataset_yolotid/1015/  --imgsz 1080 1920
```

Then it will generate the following folders under this root:
```
.
├── dataset_yolotid/
│    ├── 1015/
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

## MOT2D format

python convert_tid.py --staff_annotation_root ./staff_bbox_annotation/
--to_root ./serial_number_trackid/ __--format mot2d__ --imgsz height width 

- frame ID starts from 0

E.g. 
```
python convert_tid.py --staff_annotation_root ./dataset/1015/ --to_root ./dataset_mot2d/1015/  --imgsz 1080 1920 --format mot2d
```

Then it will generate the following folders under this root:
```
.
├── dataset_mot2d/
│    ├── 1015/
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