"""
Тестирование модели YOLOv8s на изображениях из валидационной выборки
"""

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================
# НАСТРОЙКИ
# ============================================

# Путь к модели YOLOv8s (после обучения)
MODEL_PATH = 'runs/detect/runs/train/yolov8s_30_epochs/weights/best.pt'

# Папка с изображениями для теста
TEST_IMG_DIR = 'data/raw/images/val2017/'

# Количество изображений для проверки
NUM_IMAGES = 6

# Папка для сохранения результатов
SAVE_DIR = 'results/predictions_yolov8s/'
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================

print(f"📋 Загрузка модели из {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ============================================
# ВЫБОР ИЗОБРАЖЕНИЙ
# ============================================

# Получаем список всех изображений в папке
image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Берем первые NUM_IMAGES (или все, если их меньше)
images_to_test = image_files[:NUM_IMAGES]

print(f"🖼️ Найдено {len(image_files)} изображений. Выбрано {len(images_to_test)} для проверки.")

# ============================================
# ПРЕДСКАЗАНИЯ И ВИЗУАЛИЗАЦИЯ
# ============================================

print("\n🔍 Выполнение предсказаний...")

for i, img_file in enumerate(images_to_test):
    img_path = os.path.join(TEST_IMG_DIR, img_file)
    
    # Выполняем предсказание
    results = model(img_path)
    
    # Получаем изображение с нарисованными рамками
    annotated_img = results[0].plot()
    
    # Сохраняем результат
    save_path = os.path.join(SAVE_DIR, f'pred_{img_file}')
    cv2.imwrite(save_path, annotated_img)
    print(f"   Сохранено: {save_path}")

print("\n✅ Все изображения обработаны и сохранены!")
print(f"📁 Результаты сохранены в папке: {SAVE_DIR}")

# ============================================
# ОТОБРАЖЕНИЕ ПРИМЕРА (ОПЦИОНАЛЬНО)
# ============================================

# Показать одно изображение для примера
if len(images_to_test) > 0:
    example_img_path = os.path.join(SAVE_DIR, f'pred_{images_to_test[0]}')
    if os.path.exists(example_img_path):
        example_img = cv2.imread(example_img_path)
        example_img_rgb = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(example_img_rgb)
        plt.title('Пример предсказания модели YOLOv8s')
        plt.axis('off')
        plt.show()

# ============================================
# ВЫВОД СТАТИСТИКИ ПО МОДЕЛИ
# ============================================

print("\n📊 Информация о модели:")
print(f"   Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Размер модели: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")