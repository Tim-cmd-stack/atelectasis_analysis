from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
import torch
import random

# Фиксируем глобальный seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


# Enhanced transforms with DICOM-style preprocessing
transform = transforms.Compose([
    DicomStyleImagePreprocessor(target_size=(224, 224), apply_clahe=True),  # Custom DICOM-style preprocessing
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Alternative transform without CLAHE (if you want to compare)
'''transform_no_clahe = transforms.Compose([
    DicomStyleImagePreprocessor(target_size=(224, 224), apply_clahe=False),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])'''

# Use ImageFolder with enhanced transforms
dataset = datasets.ImageFolder(
    root='C:\\Users\\CYBER ARTEL\\.cache\\kagglehub\\datasets\\nih-chest-xrays\\data\\nih_custom_dataset\\new_classes',
    transform=transform
)

# Split dataset into train and validation
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# Разделение на train/val с фиксированным seed
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=[dataset.targets[i] for i in indices],
    random_state=SEED
)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# DataLoader с фиксированным перемешиванием
g = torch.Generator()
g.manual_seed(SEED)

classification_train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    generator=g
)

classification_val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
)

if __name__ == '__main__':
    # Display class information
    print("Классы:", dataset.classes)
    print("Соответствие класс-индекс:", dataset.class_to_idx)
    # Test batch dimensions
    batch = next(iter(classification_train_loader))
    images, labels = batch
    print(f"Размерность изображений: {images.shape}")  # Should be [batch_size, channels, height, width]
    print(f"Размерность меток: {labels.shape}")  # Should be [batch_size]

    # Optional: Save a sample processed image to see the effect
    sample_image = images[0]  # Get first image from batch
    # Convert from tensor to numpy for visualization
    sample_np = sample_image.permute(1, 2, 0).numpy()
    # Denormalize for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    sample_np = sample_np * std + mean
    sample_np = np.clip(sample_np, 0, 1)


    # Method 2: Direct save using cv2 (exact pixel dimensions)
    sample_cv2 = (sample_np * 255).astype(np.uint8)
    cv2.imwrite("sample_processed_exact.png", cv2.cvtColor(sample_cv2, cv2.COLOR_RGB2BGR))

    print("Sample processed images saved:")
    print("- 'sample_processed_224x224.png' (matplotlib, should be close to 224x224)")
    print("- 'sample_processed_exact.png' (opencv, exactly 224x224)")
    print(f"Actual tensor image shape: {sample_np.shape}")  # This will show (224, 224, 3)