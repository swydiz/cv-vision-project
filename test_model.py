"""
Проверка работы обученной модели YOLO на нескольких изображениях
"""

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================
# НАСТРОЙКИ
# ============================================

# Путь к вашей лучшей модели (после обучения)
MODEL_PATH = 'runs/detect/results/train/weights/best.pt'

# Папка с изображениями для теста (возьмите несколько из валидационной выборки)
# Вы также можете использовать свои изображения
TEST_IMG_DIR = 'data/raw/images/val2017/'

# Количество случайных изображений для проверки
NUM_IMAGES = 6

# Папка для сохранения результатов
SAVE_DIR = 'results/predictions/'
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

plt.figure(figsize=(15, 10))

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
    example_img = cv2.imread(os.path.join(SAVE_DIR, f'pred_{images_to_test[0]}'))
    example_img_rgb = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(example_img_rgb)
    plt.title('Пример предсказания модели')
    plt.axis('off')
    plt.show()