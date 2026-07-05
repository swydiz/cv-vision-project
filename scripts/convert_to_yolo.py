import json
import os
from pathlib import Path

TRAIN_JSON = 'data/raw/annotations/instances_train2017_cleaned.json'
VAL_JSON = 'data/raw/annotations/instances_val2017_cleaned.json'

OUTPUT_TRAIN = 'data/processed/yolo_labels/train2017/'
OUTPUT_VAL = 'data/processed/yolo_labels/val2017/'

def convert_coco_to_yolo(json_path, output_dir):
    """Конвертирует COCO JSON в YOLO TXT файлы"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_id_to_info = {img['id']: img for img in data['images']}
    image_id_to_ann = {}
    for ann in data['annotations']:
        image_id_to_ann.setdefault(ann['image_id'], []).append(ann)
    
    category_ids = {cat['id']: idx for idx, cat in enumerate(data['categories'])}
    
    for image_id, anns in image_id_to_ann.items():
        img_info = image_id_to_info[image_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        image_name = Path(img_info['file_name']).stem
        txt_path = os.path.join(output_dir, f"{image_name}.txt")
        
        with open(txt_path, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in category_ids:
                    continue
                
                class_id = category_ids[cat_id]
                x, y, w, h = ann['bbox']
                
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    
    print(f"{output_dir}")
    print(f"Изображений: {len(image_id_to_ann)}")
    print(f"Аннотаций: {len(data['annotations'])}")

if __name__ == "__main__":
    print("Конвертация COCO - YOLO...")
    
    convert_coco_to_yolo(TRAIN_JSON, OUTPUT_TRAIN)
    convert_coco_to_yolo(VAL_JSON, OUTPUT_VAL)
    
    print("Готово")