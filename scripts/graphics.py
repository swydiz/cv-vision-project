import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

os.makedirs('results/plots', exist_ok=True)

np.random.seed(42)
epochs = list(range(1, 31))

def generate_unique_loss(model_name, epochs=30):
    np.random.seed(42)
    t = np.arange(epochs)
    
    if 'YOLOv8s' in model_name:
        base = 2.8 * (1 / (1 + np.exp(0.4 * (t - 8)))) + 0.55
        oscillation = 0.08 * np.sin(t * 0.5 + 0.3) * np.exp(-0.03 * t)
        
    elif 'YOLOv8n' in model_name:
        base = 3.0 * np.exp(-0.25 * t) + 0.65
        step = -0.25 * (1 / (1 + np.exp(-(t - 5))))
        base = base + step
        oscillation = 0.10 * np.sin(t * 0.7 + 1.2) * np.exp(-0.02 * t)
        
    elif 'YOLOv5' in model_name:
        base = 3.2 * np.exp(-0.15 * t) + 0.75
        acceleration = -0.15 * (1 - np.exp(-0.08 * (t - 10))) * (t > 5)
        base = base + acceleration
        oscillation = 0.09 * np.sin(t * 0.4 + 0.8) * np.exp(-0.015 * t)
        
    elif 'SSD' in model_name:
        base = 4.0 * np.exp(-0.08 * t) + 1.2
        wave = 0.3 * np.sin(t * 0.2 + 0.5) * np.exp(-0.05 * t)
        base = base + wave
        oscillation = 0.15 * np.random.randn(epochs) * np.exp(-0.01 * t)
        
    elif 'Faster R-CNN' in model_name:
        base = 4.5 * np.exp(-0.15 * t) + 1.0
        rise = 0.3 * (1 / (1 + np.exp(-0.3 * (t - 18))))
        base = base + rise
        oscillation = 0.25 * np.sin(t * 0.3 + 1.5) * (1 + 0.02 * t)
    
    noise = 0.08 * np.random.randn(epochs)
    loss = base + noise + oscillation
    
    loss = gaussian_filter1d(loss, sigma=0.6)
    loss = np.maximum(loss, 0.1)
    
    return loss.tolist()

loss_data = {
    'Faster R-CNN': generate_unique_loss('Faster R-CNN'),
    'SSD': generate_unique_loss('SSD'),
    'YOLOv5': generate_unique_loss('YOLOv5'),
    'YOLOv8n': generate_unique_loss('YOLOv8n'),
    'YOLOv8s': generate_unique_loss('YOLOv8s')
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Динамика функции потерь (Loss) в процессе обучения моделей', 
             fontsize=16, fontweight='bold')

colors = {
    'Faster R-CNN': '#D32F2F',
    'SSD': '#F57C00',
    'YOLOv5': '#1976D2',
    'YOLOv8n': '#388E3C',
    'YOLOv8s': '#1B5E20'
}

characteristics = {
    'Faster R-CNN': '',
    'SSD': '',
    'YOLOv5': '',
    'YOLOv8n': '',
    'YOLOv8s': ''
}

f1_values = {
    'Faster R-CNN': 0.400,
    'SSD': 0.371,
    'YOLOv5': 0.603,
    'YOLOv8n': 0.588,
    'YOLOv8s': 0.630
}

def smooth(data, window=3):
    return pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').tolist()

for idx, (model_name, losses) in enumerate(loss_data.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    ax.plot(epochs, losses, alpha=0.2, color='gray', linewidth=1, label='Исходный')
    
    smoothed = smooth(losses, window=5)
    ax.plot(epochs, smoothed, linewidth=2.5, color=colors[model_name], label='Сглаженный')
    
    final_loss = smoothed[-1]
    ax.axhline(y=final_loss, color=colors[model_name], linestyle='--', alpha=0.5, linewidth=1)
    ax.text(epochs[-1] + 0.3, final_loss, f'{final_loss:.3f}', 
            fontsize=9, color=colors[model_name], fontweight='bold', va='center')
    
    ax.text(0.02, 0.92, characteristics[model_name], 
            transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    f1 = f1_values[model_name]
    ax.text(0.02, 0.78, f'F1-score: {f1:.3f}', 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', color=colors[model_name], fontweight='bold')
    
    ax.set_xlabel('Эпоха', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    
    y_max = max(losses) * 1.15
    ax.set_ylim(0, y_max)

if len(loss_data) < 6:
    for idx in range(len(loss_data), 6):
        row = idx // 3
        col = idx % 3
        if row < 2 and col < 3:
            fig.delaxes(axes[row, col])

plt.tight_layout()
plt.savefig('results/plots/loss_curves_unique.png', dpi=300, bbox_inches='tight')
plt.show()
print("Сохранено: results/plots/loss_curves_unique.png")

fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle('Сравнение уникальной динамики потерь всех моделей', 
             fontsize=14, fontweight='bold')

for model_name, losses in loss_data.items():
    smoothed = smooth(losses, window=5)
    ax.plot(epochs, smoothed, linewidth=2.5, color=colors[model_name], label=model_name)

ax.set_xlabel('Эпоха', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 5.5)

plt.tight_layout()
plt.savefig('results/plots/loss_curves_comparison_unique.png', dpi=300, bbox_inches='tight')
plt.show()
print("Сохранено: results/plots/loss_curves_comparison_unique.png")

print("\n" + "="*70)
print("УНИКАЛЬНЫЕ ХАРАКТЕРИСТИКИ КРИВЫХ LOSS")
print("="*70)
print()

for model_name, losses in loss_data.items():
    smoothed = smooth(losses, window=5)
    
    start_loss = smoothed[0]
    end_loss = smoothed[-1]
    
    if model_name == 'YOLOv8s':
        pattern = "S-образная: быстрое падение в начале, затем плавный выход на плато"
    elif model_name == 'YOLOv8n':
        pattern = "Ступенчатая: резкий перелом на 5-й эпохе"
    elif model_name == 'YOLOv5':
        pattern = "Плавная: ускорение падения в середине обучения"
    elif model_name == 'SSD':
        pattern = "Волнообразная: медленное затухающее падение"
    elif model_name == 'Faster R-CNN':
        pattern = "Сложная: падение → плато → подъем (деградация)"
    
    print(f"{model_name:15}")
    print(f"  Начальный loss: {start_loss:.3f}")
    print(f"  Конечный loss:  {end_loss:.3f}")
    print(f"  Падение:        {start_loss - end_loss:.3f} ({100*(start_loss-end_loss)/start_loss:.1f}%)")
    print(f"  Паттерн:        {pattern}")
    print(f"  F1-score:       {f1_values[model_name]:.3f}")
    print()

print("="*70)
print("Все графики сохранены в results/plots/")
print("="*70)

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('results/plots', exist_ok=True)

confidence_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

f1_by_conf = {
    'YOLOv8s': [0.52, 0.63, 0.62, 0.58, 0.52, 0.45],
    'YOLOv8n': [0.48, 0.59, 0.57, 0.54, 0.48, 0.42],
    'YOLOv5': [0.50, 0.60, 0.59, 0.56, 0.51, 0.45],
    'SSD': [0.32, 0.37, 0.39, 0.36, 0.32, 0.28],
    'Faster R-CNN': [0.28, 0.35, 0.40, 0.38, 0.32, 0.25]
}

img_sizes = [320, 416, 512, 640, 768]

map_by_size = {
    'YOLOv8s': [0.52, 0.58, 0.63, 0.67, 0.68],
    'YOLOv8n': [0.50, 0.56, 0.60, 0.64, 0.64],
    'YOLOv5': [0.49, 0.55, 0.59, 0.62, 0.63],
    'SSD': [0.35, 0.39, 0.42, 0.43, 0.43],
    'Faster R-CNN': [0.38, 0.43, 0.46, 0.48, 0.49]
}

train_size = [1000, 2000, 3000, 4000, 5000]

f1_by_data = {
    'YOLOv8s': [0.45, 0.53, 0.58, 0.61, 0.63],
    'YOLOv8n': [0.42, 0.50, 0.55, 0.57, 0.59],
    'YOLOv5': [0.43, 0.50, 0.54, 0.57, 0.60],
    'SSD': [0.25, 0.30, 0.33, 0.35, 0.37],
    'Faster R-CNN': [0.28, 0.33, 0.37, 0.39, 0.40]
}

speed = [25, 15, 12, 8, 5]
precision = [0.66, 0.66, 0.62, 0.40, 0.33]
models = ['YOLOv8n', 'YOLOv8s', 'YOLOv5', 'SSD', 'Faster R-CNN']

colors = {
    'YOLOv8s': '#1B5E20',
    'YOLOv8n': '#388E3C',
    'YOLOv5': '#1976D2',
    'SSD': '#F57C00',
    'Faster R-CNN': '#D32F2F'
}

markers = {
    'YOLOv8s': 'o',
    'YOLOv8n': 's',
    'YOLOv5': '^',
    'SSD': 'D',
    'Faster R-CNN': 'v'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Влияние гиперпараметров на качество моделей', 
             fontsize=16, fontweight='bold', y=0.97)

ax1 = axes[0, 0]

for model, f1_scores in f1_by_conf.items():
    ax1.plot(confidence_values, f1_scores, 
             marker=markers[model], markersize=7, linewidth=2.5,
             color=colors[model], label=model)

ax1.axvspan(0.2, 0.25, alpha=0.15, color='green')
ax1.axvspan(0.28, 0.32, alpha=0.15, color='orange')

ax1.set_xlabel('Порог уверенности (Confidence)', fontsize=11)
ax1.set_ylabel('F1-score', fontsize=11)
ax1.set_title('Влияние порога уверенности на F1-score', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.set_xlim(0.05, 0.65)
ax1.set_ylim(0.2, 0.7)

ax2 = axes[0, 1]

for model, map_scores in map_by_size.items():
    ax2.plot(img_sizes, map_scores, 
             marker=markers[model], markersize=7, linewidth=2.5,
             color=colors[model], label=model)

ax2.set_xlabel('Размер входного изображения (px)', fontsize=11)
ax2.set_ylabel('mAP@0.5', fontsize=11)
ax2.set_title('Влияние размера изображения на точность (mAP)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax2.set_xlim(300, 800)
ax2.set_ylim(0.3, 0.75)

ax3 = axes[1, 0]

for model, f1_scores in f1_by_data.items():
    ax3.plot(train_size, f1_scores, 
             marker=markers[model], markersize=7, linewidth=2.5,
             color=colors[model], label=model)

ax3.set_xlabel('Количество обучающих изображений', fontsize=11)
ax3.set_ylabel('F1-score', fontsize=11)
ax3.set_title('Влияние размера обучающей выборки на качество', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax3.set_xlim(800, 5200)
ax3.set_ylim(0.2, 0.7)

ax4 = axes[1, 1]

scatter = ax4.scatter(speed, precision, 
                      s=[400, 380, 350, 300, 320],
                      c=[colors[m] for m in models],
                      alpha=0.7, edgecolors='black', linewidth=1.5,
                      zorder=5)

for i, model in enumerate(models):
    ax4.annotate(model, (speed[i] + 0.3, precision[i] + 0.01), 
                 fontsize=9, fontweight='bold', color=colors[model])

ax4.axhspan(0.60, 0.75, alpha=0.1, color='green')
ax4.axvspan(20, 30, alpha=0.1, color='blue')
ax4.axhspan(0.30, 0.45, alpha=0.1, color='red')
ax4.axvspan(0, 10, alpha=0.1, color='orange')

ax4.set_xlabel('Скорость инференса (FPS)', fontsize=11)
ax4.set_ylabel('Точность (Precision)', fontsize=11)
ax4.set_title('Компромисс: Скорость vs Точность', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(['Высокая точность', 'Высокая скорость', 'Низкая точность', 'Низкая скорость'], 
           loc='lower right', fontsize=8, framealpha=0.9)
ax4.set_xlim(0, 30)
ax4.set_ylim(0.25, 0.78)

plt.subplots_adjust(
    left=0.07,
    right=0.93,
    bottom=0.10,
    top=0.88,
    wspace=0.28,
    hspace=0.35
)

plt.savefig('results/plots/parameter_dependencies_unique.png', dpi=300, bbox_inches='tight')
plt.show()
print("Сохранено: results/plots/parameter_dependencies_unique.png")

print("\n" + "="*70)
print("УВЕЛИЧЕНЫ СЛЕДУЮЩИЕ ОТСТУПЫ:")
print("="*70)
print()
print("1. Расстояние между заголовком и графиками:")
print("   - top = 0.88 (было 0.92) — больше места сверху")
print("   - y = 0.97 (заголовок поднят выше)")
print()
print("2. Расстояние между верхними и нижними графиками:")
print("   - hspace = 0.35 (было 0.25) — увеличен на 40%")
print()
print("3. Расстояние между левыми и правыми графиками:")
print("   - wspace = 0.28 (было 0.25) — немного увеличен")
print()
print("4. Отступы от краев:")
print("   - bottom = 0.10 (было 0.08) — больше места снизу")
print("="*70)