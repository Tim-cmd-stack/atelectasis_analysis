import pydicom
import numpy as np
import cv2
import datetime
import os
import json
from PIL import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


class DicomProcessor:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.anonymization_log = os.path.join(self.log_dir, "anonymization.log")

    def anonymize_dicom(self, ds: pydicom.Dataset) -> tuple:
        """Анонимизирует DICOM файл"""
        removed_data = {}
        sensitive_tags = [
            (0x0010, 0x0010),  # PatientName
            (0x0010, 0x0030),  # PatientBirthDate
            (0x0010, 0x0040),  # PatientSex
            (0x0008, 0x0080),  # InstitutionName
        ]

        for tag in sensitive_tags:
            if tag in ds:
                tag_name = pydicom.datadict.keyword_for_tag(tag)
                removed_data[tag_name] = str(ds[tag].value)
                del ds[tag]

        # Логирование - FIXED: Use datetime.datetime.now()
        if hasattr(ds, 'StudyInstanceUID'):
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "study_instance_uid": str(ds.StudyInstanceUID),
                "removed_fields": list(removed_data.keys())
            }
            with open(self.anonymization_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return ds, removed_data

    def preprocess_dicom(self, dicom_path: str, target_size=(224, 224)) -> tuple:
        """Предобрабатывает DICOM файл"""
        ds = pydicom.dcmread(dicom_path)

        # Сохраняем метаданные
        metadata = {
            "StudyInstanceUID": str(ds.StudyInstanceUID) if hasattr(ds, 'StudyInstanceUID') else None,
            "SeriesInstanceUID": str(ds.SeriesInstanceUID) if hasattr(ds, 'SeriesInstanceUID') else None,
            "SOPInstanceUID": str(ds.SOPInstanceUID) if hasattr(ds, 'SOPInstanceUID') else None,
            "Modality": str(ds.Modality) if hasattr(ds, 'Modality') else None,
            "ViewPosition": str(ds.ViewPosition) if hasattr(ds, 'ViewPosition') else None,
        }

        # Анонимизация
        ds, removed_data = self.anonymize_dicom(ds)
        metadata["anonymized_data"] = removed_data

        # Извлечение и обработка изображения
        img = ds.pixel_array.astype(np.float32)

        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept

        # Нормализация
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
        img = cv2.resize(img, target_size)

        # Преобразование к RGB
        img_rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)

        return img_rgb, metadata, ds

    def create_dicom_from_png(self, png_path, output_dcm_path, add_patient_info=True):
        """Создает DICOM из PNG (для тестирования)"""
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {png_path}")

        img = cv2.resize(img, (1024, 1024))

        # File meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Dataset
        ds = FileDataset(output_dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

        if add_patient_info:
            ds.PatientName = "Test^Patient^Name"
            ds.PatientID = "123456789"
            ds.PatientBirthDate = "19800101"
            ds.PatientSex = "M"
            ds.InstitutionName = "Test Hospital"

        ds.Modality = "DX"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        # Временные метки - FIXED: Use datetime.datetime.now()
        dt = datetime.datetime.now()
        ds.StudyDate = dt.strftime('%Y%m%d')
        ds.StudyTime = dt.strftime('%H%M%S.%f')[:-3]
        ds.SeriesDate = ds.StudyDate
        ds.SeriesTime = ds.StudyTime

        # Дополнительные теги
        ds.StudyDescription = "Chest X-Ray"
        ds.SeriesDescription = "PA View"
        ds.ViewPosition = "PA"
        ds.BodyPartExamined = "CHEST"

        # Параметры изображения
        ds.Rows, ds.Columns = img.shape
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1

        # Конвертируем изображение
        img_16bit = (img.astype(np.float32) / 255.0 * 4095).astype(np.uint16)
        ds.PixelData = img_16bit.tobytes()

        ds.save_as(output_dcm_path)
        return ds