"""
Очистка данных COCO 2017
"""

import json
import os

# Пути
TRAIN_JSON = 'data/raw/annotations/instances_train2017.json'
VAL_JSON = 'data/raw/annotations/instances_val2017.json'
TRAIN_IMG_DIR = 'data/raw/train2017/'
VAL_IMG_DIR = 'data/raw/val2017/'
OUTPUT_TRAIN = 'data/raw/annotations/instances_train2017_cleaned.json'
OUTPUT_VAL = 'data/raw/annotations/instances_val2017_cleaned.json'


def clean_coco(json_path, img_dir, output_path):
    """Очистка COCO JSON"""
    
    # Загружаем
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 1. Удаляем iscrowd
    data['annotations'] = [ann for ann in data['annotations'] if ann['iscrowd'] == 0]
    
    # 2. Оставляем только изображения с аннотациями
    image_ids = set(ann['image_id'] for ann in data['annotations'])
    data['images'] = [img for img in data['images'] if img['id'] in image_ids]
    
    # 3. Проверяем, что файлы существуют
    data['images'] = [img for img in data['images'] 
                      if os.path.exists(os.path.join(img_dir, img['file_name']))]
    
    # 4. Снова фильтруем аннотации
    image_ids = set(img['id'] for img in data['images'])
    data['annotations'] = [ann for ann in data['annotations'] 
                           if ann['image_id'] in image_ids]
    
    # 5. Сохраняем
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    # Краткая статистика
    print(f"{output_path}")
    print(f"Изображений: {len(data['images'])}")
    print(f"Аннотаций: {len(data['annotations'])}")


if __name__ == "__main__":
    print("Очистка данных COCO...")
    clean_coco(TRAIN_JSON, TRAIN_IMG_DIR, OUTPUT_TRAIN)
    clean_coco(VAL_JSON, VAL_IMG_DIR, OUTPUT_VAL)
    print("Готово")