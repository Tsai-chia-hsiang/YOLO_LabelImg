import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def arg_parse():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--staff_annotation_root', type=Path, required=True, help='root contains JSON annotation files')
    parser.add_argument('--to_root', type=Path, required=True, help='Output directory for converted TXT files')
    parser.add_argument('--imgsz', type=int, nargs='+', default=(1920, 1080), help='Image size, (height, width)')
    parser.add_argument('--format', type=str, default='mot2d', help='format, yolo or mot2d')
    opt = parser.parse_args()
    
    if len(opt.imgsz) != 2 and opt.format == "yolo":
        raise ValueError("imgsz should give 2 value: w h")

    return opt
   

def createML_to_yolo(src:Path, dst:Path, img_width:int, img_height:int):
  
    """
    CreateML format to YOLO detection dataset format.

    Please Note that we also write the trackID in the end of each object notation.

    If you want to use the converted files as YOLO detection labels, please 
    remove the last value for each line to make it suitable for YOLO.

    Args
    ---
    - src: the root for containing CreateML notation file for a frame
    - dst : output dir
    - img_height, img_width : the height and width for the target frame in order to normalize bbox
    """
    
    def convert_CreateML_to_YOLO_detection_with_tid(json_path:os.PathLike, output_path:os.PathLike, img_width:int, img_height:int):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if len(data) == 0:
            return 
        
        annotations = data[0]['annotations']

        with open(output_path, 'w') as f_out:
            for annotation in annotations:
                
                coordinates = annotation['coordinates']
                label = annotation['label'] 
                
                # Extract track_ID by removing the 'stuffID_' prefix
                if "_" not in label:
                    # not a legal object notation
                    continue
            
                track_id = int(label.split('_')[1])

                # Normalize the coordinates
                center_x = coordinates['x'] / img_width
                center_y = coordinates['y'] / img_height
                width = coordinates['width'] / img_width
                height = coordinates['height'] / img_height

                f_out.write(f'0 {center_x} {center_y} {width} {height} {track_id}\n')
        
    json_annotation_files = [i for i in Path(src).glob("*.json")]
  
    for json_file in tqdm(json_annotation_files, desc='Converting JSON to TXT'):
        convert_CreateML_to_YOLO_detection_with_tid(
            json_path=json_file, output_path = dst/f"{json_file.stem}.txt", 
            img_height=img_height, img_width=img_width
        )


def createML_to_mot2d (src:Path, dst:Path, *args):

    """
    
    CreateML format to MOT2D gt.txt format.

    <frame>,<tID>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,1,-1,-1,-1

    Please note that the <frame> start from 0 but not 1 

    Args
    ---
    - src: the root for containing CreateML notation file for a frame
    - dst : output dir
    
    """
    json_annotation_files = [i for i in Path(src).glob("*.json")]
    json_annotation_files.sort(key=lambda x:int(x.stem.split("_")[-1]) )
    fout = open(dst, "w+") 
    
    for fid, json_file in enumerate(tqdm(json_annotation_files, desc='Converting JSON to TXT')):
        data = []
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if len(data) == 0:
            continue
        
        for annotation in data[0]['annotations']:
            
            coordinates = annotation['coordinates']
            label = annotation['label'] 
            
            if "_" not in label:
                # not a legal object notation
                continue
            
            track_id = int(label.split('_')[1])
            width = coordinates['width'] 
            height = coordinates['height']  
            left = coordinates['x'] - (width/2)
            top = coordinates['y'] - (height/2)      
            
            print(f"{fid}, {track_id}, {left}, {top}, {width}, {height}, 1, -1, -1, -1", file=fout)
    
    fout.close()

CONVERT_MAP = {
    'yolo':createML_to_yolo,
    'mot2d':createML_to_mot2d
}

if __name__ == '__main__':
    
    args = arg_parse()
    convert_root = Path(args.staff_annotation_root) 
    
    to_root = Path(args.to_root)
    to_root.mkdir(parents=True, exist_ok=True)
    prompt = f"convert to {args.format} on scale :{args.imgsz}" \
        if args.format == 'yolo' else f"convert to {args.format}"
    
    print(prompt)

    for r in convert_root.iterdir():
        if r.is_dir():
            srcs = [f for f in r.iterdir() if f.is_dir()]
            for si in srcs:
                # should contain only one subfolder for clip annotations
                src = [_ for _ in si.iterdir()][0]
                ci = src.parts
                dst = to_root/ci[-3]/ci[-2]/ci[-1]
                dst.mkdir(parents=True, exist_ok=True)
                
                if args.format == 'yolo':
                    dst.mkdir(parents=True, exist_ok=True)
                    print(f"{src}/*.json -> {dst}/*.txt : ")
                
                elif args.format == 'mot2d':
                    dst = dst/f'{src.name}.txt'
                    print(f"{src}/*.json -> {dst}")

                CONVERT_MAP[args.format](src,dst,args.imgsz[0],args.imgsz[1])

