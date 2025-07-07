import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from B_training_teacher_classifier import DeiTClassifierWSOL


# DicomStyleImagePreprocessor class (copied from roc_auc_comparison.py)
class DicomStyleImagePreprocessor:
    """Custom transform that applies DICOM-style preprocessing to regular images"""

    def __init__(self, target_size=(224, 224), apply_clahe=True):
        self.target_size = target_size
        self.apply_clahe = apply_clahe

    def apply_clahe_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_uint8)

    def __call__(self, pil_image):
        # Convert PIL to numpy array
        if pil_image.mode == 'RGB':
            # Convert RGB to grayscale for medical image style processing
            img = np.array(pil_image.convert('L')).astype(np.float32)
        else:
            img = np.array(pil_image).astype(np.float32)

        # Intensity normalization to 0-255 range (similar to DICOM processing)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min + 1e-8) * 255.0
        else:
            img = np.zeros_like(img)

        # Apply CLAHE for contrast enhancement
        if self.apply_clahe:
            img = self.apply_clahe_enhancement(img)

        # Resize image
        img = cv2.resize(img, self.target_size)

        # Convert grayscale to RGB by stacking channels (like in DICOM handler)
        img_rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)

        # Convert back to PIL Image for torchvision transforms
        return Image.fromarray(img_rgb)


# --- Настройки ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\best_deit_scm_model_2.pth"
image_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\new_classes\Atelectasis\00002425_003.png"

# Трансформации точно как в roc_auc_comparison.py
transform = transforms.Compose([
    DicomStyleImagePreprocessor(target_size=(224, 224), apply_clahe=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Названия классов (как в roc_auc_comparison.py)
class_names = ['Atelectasis', 'No_pathologies', 'Other_pathologies']


# --- Функция для преобразования тепловой карты в bbox ---
def heatmap_to_bbox(heatmap, threshold=0.5):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_bin = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return [x, y, x + w, y + h]


# --- Загрузка модели ---
print(f"Используется устройство: {device}")
print(f"Загрузка модели из: {checkpoint_path}")

model = DeiTClassifierWSOL(num_classes=3, pretrained=False, scm_blocks=4)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

print("Модель загружена успешно")

# --- Загрузка и обработка изображения ---
print(f"Обработка изображения: {image_path}")

# Загружаем оригинальное изображение
original_image = Image.open(image_path).convert("RGB")
print(f"Размер оригинального изображения: {original_image.size}")

# Применяем ту же предобработку, что и в roc_auc_comparison.py
preprocessed_image = DicomStyleImagePreprocessor(target_size=(224, 224), apply_clahe=True)(original_image)
print(f"Размер после предобработки: {preprocessed_image.size}")

# Применяем полные трансформации
image_tensor = transform(original_image).unsqueeze(0).to(device)
print(f"Размер тензора: {image_tensor.shape}")

# --- Инференс модели ---
with torch.no_grad():
    # Получаем предсказания классификации
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1).item()

    # Получаем тепловую карту локализации
    heatmap = model.localize(image_tensor)[0].cpu().numpy()

print(f"Размер тепловой карты: {heatmap.shape}")

# --- Результаты классификации ---
print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print("=" * 50)

print(f"Предсказанный класс: {class_names[predicted_class]}")
print(f"Вероятности по классам:")
for i, class_name in enumerate(class_names):
    prob = probabilities[0, i].item()
    print(f"  {class_name}: {prob:.4f}")

max_prob = torch.max(probabilities).item()
print(f"Максимальная вероятность: {max_prob:.4f}")

# --- Обработка тепловой карты ---
# Интерполяция тепловой карты до размера изображения
heatmap_up = F.interpolate(
    torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
    size=(224, 224),
    mode='bilinear', align_corners=False
)[0, 0].numpy()

print(f"Размер интерполированной тепловой карты: {heatmap_up.shape}")

# --- Получение bbox ---
threshold = 0.4
predicted_bbox = heatmap_to_bbox(heatmap_up, threshold=threshold)

if predicted_bbox:
    print(f"Предсказанный bbox (порог {threshold}): {predicted_bbox}")
else:
    print(f"Bbox не найден с порогом {threshold}")

# --- Визуализация ---
print("\nСоздание визуализации...")

# Подготавливаем изображение для визуализации (используем предобработанное)
img_vis = np.array(preprocessed_image).copy()

# Создаем оверлей из тепловой карты
overlay = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min() + 1e-8)
overlay_colored = cv2.applyColorMap(np.uint8(255 * overlay), cv2.COLORMAP_JET)

# Смешиваем изображение с тепловой картой
overlayed = cv2.addWeighted(img_vis, 0.6, overlay_colored, 0.4, 0)

# Рисуем bbox если он найден
if predicted_bbox:
    cv2.rectangle(overlayed, tuple(predicted_bbox[:2]), tuple(predicted_bbox[2:]), (0, 255, 0), 2)

# --- Создаем комплексную визуализацию ---
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Оригинальное изображение
axes[0, 0].imshow(original_image)
axes[0, 0].set_title("Оригинальное изображение")
axes[0, 0].axis("off")

# Предобработанное изображение
axes[0, 1].imshow(preprocessed_image)
axes[0, 1].set_title("После DICOM-стиль предобработки")
axes[0, 1].axis("off")

# Тепловая карта
axes[1, 0].imshow(heatmap_up, cmap='jet')
axes[1, 0].set_title("Тепловая карта локализации")
axes[1, 0].axis("off")

# Финальный результат с наложением
axes[1, 1].imshow(overlayed[..., ::-1])  # OpenCV BGR -> RGB
title = f"Результат: {class_names[predicted_class]} ({max_prob:.3f})"
if predicted_bbox:
    title += f"\nBBox: {predicted_bbox}"
axes[1, 1].set_title(title)
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()

# --- Дополнительная информация ---
print(f"\n" + "=" * 50)
print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
print("=" * 50)

print(f"Применена CLAHE предобработка: Да")
print(f"Размер входного изображения для модели: 224x224")
print(f"Нормализация: ImageNet стандартная")
print(f"Устройство обработки: {device}")

# Статистика по тепловой карте
print(f"\nСтатистика тепловой карты:")
print(f"  Минимальное значение: {heatmap_up.min():.4f}")
print(f"  Максимальное значение: {heatmap_up.max():.4f}")
print(f"  Среднее значение: {heatmap_up.mean():.4f}")
print(f"  Стандартное отклонение: {heatmap_up.std():.4f}")

# Проверяем уверенность модели
if max_prob < 0.5:
    print(f"\n⚠️  ВНИМАНИЕ: Низкая уверенность модели ({max_prob:.3f})")
elif max_prob < 0.7:
    print(f"\n⚠️  Умеренная уверенность модели ({max_prob:.3f})")
else:
    print(f"\n✅ Высокая уверенность модели ({max_prob:.3f})")

print(f"\nПредобработка изображения соответствует roc_auc_comparison.py")