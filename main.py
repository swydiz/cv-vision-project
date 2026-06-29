import json
from collections import Counter
import matplotlib.pyplot as plt

# Загружаем данные
with open('data/raw/annotations/instances_train2017.json', 'r') as f:
    coco_data = json.load(f)

# Считаем объекты по классам
category_counts = Counter()
for ann in coco_data['annotations']:
    category_counts[ann['category_id']] += 1

# Создаем словарь ID → название
id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

# Сортируем по убыванию
sorted_categories = category_counts.most_common()

# Выводим топ-10
print("\n=== Топ-10 самых частых классов ===")
for cat_id, count in sorted_categories[:10]:
    print(f"{id_to_name[cat_id]}: {count} объектов")

# Выводим топ-10 самых редких
print("\n=== Топ-10 самых редких классов ===")
for cat_id, count in sorted_categories[-10:]:
    print(f"{id_to_name[cat_id]}: {count} объектов")

# Визуализируем распределение (топ-20)
top_20 = sorted_categories[:20]
names = [id_to_name[cat_id] for cat_id, _ in top_20]
counts = [count for _, count in top_20]

plt.figure(figsize=(14, 8))
plt.barh(names, counts)
plt.xlabel('Количество объектов')
plt.title('Топ-20 классов в COCO 2017 (обучающая выборка)')
plt.tight_layout()
plt.savefig('coco_class_distribution.png', dpi=150)
plt.show()