import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import json
from torchvision import transforms
from datetime import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import pydicom.uid

# Импорт модели (предполагается, что она доступна)
from Neural_nets_training.B_training_teacher_classifier import DeiTClassifierWSOL


class AtelectasisDetector:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.manufacturer = "AtelectasisAI"
        self.software_version = "1.0"
        self.observer_type = "AI"

        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self._load_model()

    def _load_model(self):
        """Загружает модель"""
        model = DeiTClassifierWSOL(num_classes=3, pretrained=False, scm_blocks=4)
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def heatmap_to_bbox(self, heatmap, threshold=0.5, original_size=(1024, 1024)):
        """Преобразует тепловую карту в bbox"""
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_bin = (heatmap > threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        scale_x = original_size[0] / heatmap.shape[1]
        scale_y = original_size[1] / heatmap.shape[0]

        return [int(x * scale_x), int(y * scale_y), int((x + w) * scale_x), int((y + h) * scale_y)]

    def classify_pathology(self, probabilities):
        """Классифицирует патологию"""
        atelectasis_prob = probabilities[0][0].item()
        normal_prob = probabilities[0][1].item()
        other_pathologies_prob = probabilities[0][2].item()

        atelectasis_str = f"{atelectasis_prob * 100:.1f}%"

        if normal_prob >= 0.9:
            status = "normal"
            conclusion = f"Признаков ателектаза не обнаружено. Вероятность: {atelectasis_str}"
            other_pathologies = []
        elif atelectasis_prob >= 0.7:
            status = "atelectasis_only"
            conclusion = f"Обнаружены признаки ателектаза (вероятность: {atelectasis_str}). Требуется подтверждение врача."
            other_pathologies = []
        elif other_pathologies_prob >= 0.3:
            status = "other_pathologies"
            other_pathologies = ["pleural_effusion", "pneumothorax"][:2]
            conclusion = f"Обнаружены другие патологии. Вероятность ателектаза: {atelectasis_str}. Требуется подтверждение врача."
        else:
            status = "normal"
            conclusion = f"Признаков ателектаза не обнаружено. Вероятность: {atelectasis_str}"
            other_pathologies = []

        return {
            "status": status,
            "atelectasis_probability": atelectasis_prob,
            "atelectasis_probability_str": atelectasis_str,
            "conclusion": conclusion,
            "other_pathologies": other_pathologies
        }

    def analyze_image(self, img_rgb, original_size=(1024, 1024)):
        """Анализирует изображение"""
        img_pil = Image.fromarray(img_rgb)
        image_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            heatmap = self.model.localize(image_tensor)[0].cpu().numpy()

        # Интерполяция тепловой карты
        heatmap_up = F.interpolate(
            torch.tensor(heatmap).unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='bilinear', align_corners=False
        )[0, 0].numpy()

        # Классификация
        classification_result = self.classify_pathology(probabilities)

        # Получение bbox и локализации
        bbox = None
        location = None
        if classification_result["status"] == "atelectasis_only":
            bbox = self.heatmap_to_bbox(heatmap_up, threshold=0.4, original_size=original_size)
            if bbox:
                x_center = (bbox[0] + bbox[2]) / 2 / original_size[0]
                y_center = (bbox[1] + bbox[3]) / 2 / original_size[1]

                side = "left" if x_center < 0.5 else "right"
                if y_center < 0.33:
                    zone = "upper"
                elif y_center < 0.66:
                    zone = "middle"
                else:
                    zone = "lower"
                location = f"{zone} zone, {side} lung"

        return {
            **classification_result,
            "bbox": bbox if bbox else [],
            "location": location,
            "warning": "Заключение сгенерировано ИИ. Требуется подтверждение врача."
        }

    def create_dicom_sr(self, analysis_result, original_ds, output_path):
        """Создает DICOM SR отчет"""
        # Проверка обязательных полей
        required_fields = ['atelectasis_probability', 'bbox', 'conclusion']
        for field in required_fields:
            if field not in analysis_result:
                raise ValueError(f"Отсутствует обязательное поле: {field}")

        # File meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Main dataset
        ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SpecificCharacterSet = 'ISO_IR 192'

        # Patient Module
        ds.PatientName = getattr(original_ds, 'PatientName', "Anonymous^Patient")
        ds.PatientID = getattr(original_ds, 'PatientID', "ANON_ID_001")

        # Study Module
        ds.StudyInstanceUID = getattr(original_ds, 'StudyInstanceUID', generate_uid())
        ds.StudyDate = getattr(original_ds, 'StudyDate', datetime.now().strftime('%Y%m%d'))
        ds.StudyTime = getattr(original_ds, 'StudyTime', datetime.now().strftime('%H%M%S'))

        # SR Document Series Module
        ds.Modality = "SR"
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 999
        ds.SeriesDate = datetime.now().strftime('%Y%m%d')
        ds.SeriesTime = datetime.now().strftime('%H%M%S')

        # Equipment Module
        ds.Manufacturer = self.manufacturer
        ds.SoftwareVersions = self.software_version

        # SR Document General Module
        ds.ContentDate = datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.now().strftime('%H%M%S')
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"
        ds.ObserverType = self.observer_type

        # SOP Common Module
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        # Content Sequence
        content_sequence = []

        # Root container
        root_container = Dataset()
        root_container.RelationshipType = "CONTAINS"
        root_container.ValueType = "CONTAINER"
        root_container.ContinuityOfContent = "SEPARATE"
        root_container.ContentSequence = []

        # Findings
        findings_item = Dataset()
        findings_item.RelationshipType = "CONTAINS"
        findings_item.ValueType = "TEXT"
        findings_item.TextValue = analysis_result['conclusion']
        root_container.ContentSequence.append(findings_item)

        # Probability
        prob_item = Dataset()
        prob_item.RelationshipType = "CONTAINS"
        prob_item.ValueType = "NUM"
        measured_value = Dataset()
        measured_value.NumericValue = f"{analysis_result['atelectasis_probability']:.6f}"
        prob_item.MeasuredValueSequence = [measured_value]
        root_container.ContentSequence.append(prob_item)

        # Spatial coordinates if bbox exists
        if analysis_result['bbox'] and len(analysis_result['bbox']) == 4:
            coord_item = Dataset()
            coord_item.RelationshipType = "CONTAINS"
            coord_item.ValueType = "SCOORD"
            coord_item.GraphicType = "POLYLINE"

            x_min, y_min, x_max, y_max = analysis_result['bbox']
            coord_item.GraphicData = [
                float(x_min), float(y_min),
                float(x_max), float(y_min),
                float(x_max), float(y_max),
                float(x_min), float(y_max),
                float(x_min), float(y_min)
            ]
            root_container.ContentSequence.append(coord_item)

        content_sequence.append(root_container)
        ds.ContentSequence = content_sequence

        ds.save_as(output_path)
        return True