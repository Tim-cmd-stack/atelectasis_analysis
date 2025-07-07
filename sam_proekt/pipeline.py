import os
import json
from datetime import datetime
from dicom_processor import DicomProcessor
from ai_detector import AtelectasisDetector
import pydicom


class DicomSRGenerator:
    def __init__(self):
        self.manufacturer = "AtelectasisAI"
        self.software_version = "1.0"
        self.observer_type = "AI"
        self.disclaimer = "Для поддержки принятия врачебных решений. Не заменяет консультацию специалиста. Использование в исследовательских целях."
        self.warning = "Заключение сгенерировано ИИ. Требуется подтверждение врача."

    def validate_required_fields(self, report_data):
        """
        Проверяет наличие всех обязательных полей согласно требованиям
        """
        required_fields = {
            'atelectasis_probability': 'Вероятность ателектаза (0-1)',
            'bbox': 'Локализация (xmin, ymin, xmax, ymax)',
            'conclusion': 'Текстовое заключение'
        }

        missing_fields = []

        for field, description in required_fields.items():
            if field not in report_data or report_data[field] is None:
                missing_fields.append(f"{field} ({description})")

        # Специальная проверка для bbox
        if 'bbox' in report_data and report_data['bbox']:
            if not isinstance(report_data['bbox'], list) or len(report_data['bbox']) != 4:
                missing_fields.append("bbox (должен содержать 4 координаты: xmin, ymin, xmax, ymax)")

        # Проверка вероятности
        if 'atelectasis_probability' in report_data:
            prob = report_data['atelectasis_probability']
            if not isinstance(prob, (int, float)) or not (0 <= prob <= 1):
                missing_fields.append("atelectasis_probability (должна быть числом от 0 до 1)")

        return missing_fields

    def create_basic_sr_dataset(self, original_ds, report_data):
        """
        Создает базовый DICOM SR dataset с обязательными полями согласно требованиям
        """
        from pydicom import Dataset
        from pydicom.dataset import FileDataset
        from pydicom.uid import generate_uid
        import pydicom.uid

        # Проверяем обязательные поля
        missing_fields = self.validate_required_fields(report_data)
        if missing_fields:
            raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing_fields)}")

        # File meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Main dataset
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Устанавливаем кодировку для поддержки русского языка
        ds.SpecificCharacterSet = 'ISO_IR 192'  # UTF-8

        # Patient Module - анонимные данные, если не найдены в оригинале
        ds.PatientName = getattr(original_ds, 'PatientName', "Anonymous^Patient")
        ds.PatientID = getattr(original_ds, 'PatientID', "ANON_ID_001")

        if hasattr(original_ds, 'PatientBirthDate'):
            ds.PatientBirthDate = original_ds.PatientBirthDate
        if hasattr(original_ds, 'PatientSex'):
            ds.PatientSex = original_ds.PatientSex

        # General Study Module
        ds.StudyInstanceUID = getattr(original_ds, 'StudyInstanceUID', generate_uid())
        ds.StudyDate = getattr(original_ds, 'StudyDate', datetime.now().strftime('%Y%m%d'))
        ds.StudyTime = getattr(original_ds, 'StudyTime', datetime.now().strftime('%H%M%S'))
        ds.ReferringPhysicianName = ""
        ds.StudyID = getattr(original_ds, 'StudyID', "1")
        ds.AccessionNumber = getattr(original_ds, 'AccessionNumber', "")

        # SR Document Series Module
        ds.Modality = "SR"  # ОБЯЗАТЕЛЬНОЕ: Structured Report
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 999
        ds.SeriesDate = datetime.now().strftime('%Y%m%d')
        ds.SeriesTime = datetime.now().strftime('%H%M%S')
        ds.SeriesDescription = "AI Atelectasis Analysis Report"

        # General Equipment Module
        ds.Manufacturer = self.manufacturer
        ds.ManufacturerModelName = "AI Atelectasis Detector"
        ds.SoftwareVersions = self.software_version
        ds.DeviceSerialNumber = "AI-DETECT-001"

        # SR Document General Module
        ds.ContentDate = datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.now().strftime('%H%M%S.%f')[:-3]
        ds.InstanceNumber = 1
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"

        # Document Title
        ds.ConceptNameCodeSequence = [self._create_code("18748-4", "LN", "Diagnostic imaging report")]

        # SOP Common Module
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        # ОБЯЗАТЕЛЬНОЕ ПОЛЕ: ObserverType
        ds.ObserverType = self.observer_type

        # Referenced Image (ссылка на оригинальное изображение)
        if hasattr(original_ds, 'SOPInstanceUID'):
            self._add_referenced_image(ds, original_ds)

        return ds

    def _add_referenced_image(self, ds, original_ds):
        """
        Добавляет ссылку на оригинальное изображение
        """
        from pydicom import Dataset
        from pydicom.uid import generate_uid

        ref_image = Dataset()
        ref_image.ReferencedSOPClassUID = getattr(original_ds, 'SOPClassUID', '1.2.840.10008.5.1.4.1.1.1.1')
        ref_image.ReferencedSOPInstanceUID = original_ds.SOPInstanceUID

        ds.CurrentRequestedProcedureEvidenceSequence = [Dataset()]
        ds.CurrentRequestedProcedureEvidenceSequence[0].StudyInstanceUID = ds.StudyInstanceUID
        ds.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence = [Dataset()]
        ds.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0].SeriesInstanceUID = getattr(
            original_ds, 'SeriesInstanceUID', generate_uid()
        )
        ds.CurrentRequestedProcedureEvidenceSequence[0].ReferencedSeriesSequence[0].ReferencedImageSequence = [
            ref_image]

    def _create_code(self, code_value, coding_scheme, code_meaning):
        """
        Создает стандартный код для DICOM SR
        """
        from pydicom import Dataset

        code_item = Dataset()
        code_item.CodeValue = code_value
        code_item.CodingSchemeDesignator = coding_scheme
        code_item.CodeMeaning = code_meaning
        return code_item

    def _create_text_content(self, relationship_type, concept_code, text_value):
        """
        Создает текстовый элемент контента с проверкой длины
        """
        from pydicom import Dataset

        content_item = Dataset()
        content_item.RelationshipType = relationship_type
        content_item.ValueType = "TEXT"
        content_item.ConceptNameCodeSequence = [concept_code]

        # DICOM ограничение для TextValue (максимум 1024 символа)
        if len(text_value) > 1024:
            text_value = text_value[:1021] + "..."

        content_item.TextValue = text_value
        return content_item

    def _create_num_content(self, relationship_type, concept_code, numeric_value, unit_code=None):
        """
        Создает числовой элемент контента с проверкой формата
        """
        from pydicom import Dataset

        content_item = Dataset()
        content_item.RelationshipType = relationship_type
        content_item.ValueType = "NUM"
        content_item.ConceptNameCodeSequence = [concept_code]

        # Создаем последовательность измеренных значений
        measured_value = Dataset()

        # Форматируем числовое значение для DS VR (максимум 16 символов)
        if isinstance(numeric_value, float):
            numeric_str = f"{numeric_value:.6f}".rstrip('0').rstrip('.')
        else:
            numeric_str = str(numeric_value)

        if len(numeric_str) > 16:
            numeric_str = f"{float(numeric_value):.4f}"

        measured_value.NumericValue = numeric_str

        if unit_code:
            measured_value.MeasurementUnitsCodeSequence = [unit_code]

        content_item.MeasuredValueSequence = [measured_value]
        return content_item

    def _create_spatial_coordinates(self, bbox, reference_uid=None):
        """
        ОБЯЗАТЕЛЬНОЕ ПОЛЕ: Создает пространственные координаты для bbox в DICOM-координатах
        Формат: [xmin, ymin, xmax, ymax] в пикселях
        """
        from pydicom import Dataset

        if not bbox or len(bbox) != 4:
            raise ValueError("BoundingBox должен содержать 4 координаты: [xmin, ymin, xmax, ymax]")

        # Проверяем, что координаты валидны
        x_min, y_min, x_max, y_max = bbox
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"Некорректные координаты bbox: xmin({x_min}) >= xmax({x_max}) или ymin({y_min}) >= ymax({y_max})")

        coord_item = Dataset()
        coord_item.RelationshipType = "CONTAINS"
        coord_item.ValueType = "SCOORD"
        coord_item.ConceptNameCodeSequence = [self._create_code("111030", "DCM", "Image Region")]

        # Графический тип - прямоугольник (замкнутый полигон)
        coord_item.GraphicType = "POLYLINE"

        # Координаты прямоугольника в DICOM координатах (порядок: x1,y1, x2,y1, x2,y2, x1,y2, x1,y1)
        coord_item.GraphicData = [
            float(x_min), float(y_min),  # Левый верхний угол
            float(x_max), float(y_min),  # Правый верхний угол
            float(x_max), float(y_max),  # Правый нижний угол
            float(x_min), float(y_max),  # Левый нижний угол
            float(x_min), float(y_min)  # Замыкание контура
        ]

        # Добавляем ссылку на изображение
        if reference_uid:
            coord_item.ReferencedImageSequence = [Dataset()]
            coord_item.ReferencedImageSequence[0].ReferencedSOPInstanceUID = reference_uid

        return coord_item

    def _create_bbox_text_description(self, bbox):
        """
        Создает текстовое описание координат bounding box
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        description = f"Локализация в DICOM-координатах: xmin={x_min}, ymin={y_min}, xmax={x_max}, ymax={y_max} (размер: {width}x{height} пикселей)"
        return description

    def add_content_sequence(self, ds, report_data, original_ds):
        """Добавляет содержимое отчета в DICOM SR с русскими локализациями"""
        from pydicom import Dataset

        content_sequence = []

        # Создаем корневой контейнер
        root_container = Dataset()
        root_container.RelationshipType = "CONTAINS"
        root_container.ValueType = "CONTAINER"
        root_container.ConceptNameCodeSequence = [self._create_code("18748-4", "LN", "Диагностический отчет")]
        root_container.ContinuityOfContent = "SEPARATE"
        root_container.ContentSequence = []

        # 1. Статус анализа (переведенный на русский)
        status_translation = {
            'normal': 'Норма',
            'atelectasis_only': 'Ателектаз',
            'other_pathologies': 'Другие патологии'
        }
        status_text = status_translation.get(report_data['status'], report_data['status'])

        status_item = self._create_text_content(
            "CONTAINS",
            self._create_code("33999-4", "LN", "Статус"),
            status_text
        )
        root_container.ContentSequence.append(status_item)

        # 2. Вероятность ателектаза
        probability_item = self._create_num_content(
            "CONTAINS",
            self._create_code("113011", "DCM", "Вероятность ателектаза"),
            report_data['atelectasis_probability'],
            self._create_code("1", "UCUM", "безразмерная величина")
        )
        root_container.ContentSequence.append(probability_item)

        # 3. Заключение с предупреждением
        conclusion_text = f"{report_data['conclusion']} {self.warning}"
        conclusion_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121071", "DCM", "Заключение"),
            conclusion_text
        )
        root_container.ContentSequence.append(conclusion_item)

        # 4. Локализация (если есть, с переводом на русский)
        if report_data.get('location'):
            location_translation = {
                'upper lobe, right lung': 'Верхняя доля правого легкого',
                'middle lobe, right lung': 'Средняя доля правого легкого',
                'lower lobe, right lung': 'Нижняя доля правого легкого',
                'upper lobe, left lung': 'Верхняя доля левого легкого',
                'lower lobe, left lung': 'Нижняя доля левого легкого',
                'middle zone, right lung': 'Средняя зона правого легкого',
                'middle zone, left lung': 'Средняя зона левого легкого',
                'lower zone, right lung': 'Нижняя зона правого легкого',
                'lower zone, left lung': 'Нижняя зона левого легкого'
            }

            russian_location = location_translation.get(
                report_data['location'].lower(),
                report_data['location']
            )

            location_item = self._create_text_content(
                "CONTAINS",
                self._create_code("363698007", "SCT", "Локализация"),
                russian_location
            )
            root_container.ContentSequence.append(location_item)

        # 5. Другие патологии (если есть)
        if report_data.get('other_pathologies'):
            pathologies_text = ", ".join(report_data['other_pathologies'])
            pathologies_item = self._create_text_content(
                "CONTAINS",
                self._create_code("363698007", "SCT", "Другие патологии"),
                pathologies_text
            )
            root_container.ContentSequence.append(pathologies_item)

        # 6. Bounding box (если есть)
        if report_data.get('bbox') and len(report_data['bbox']) == 4:
            bbox_description = self._create_bbox_text_description(report_data['bbox'])
            bbox_text_item = self._create_text_content(
                "CONTAINS",
                self._create_code("111001", "DCM", "Координаты области"),
                bbox_description
            )
            root_container.ContentSequence.append(bbox_text_item)

            # Графическое представление bbox
            reference_uid = getattr(original_ds, 'SOPInstanceUID', None)
            bbox_spatial_item = self._create_spatial_coordinates(report_data['bbox'], reference_uid)
            root_container.ContentSequence.append(bbox_spatial_item)

        # 7. Предупреждение и отказ от ответственности
        warning_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121130", "DCM", "Предупреждение"),
            self.warning
        )
        root_container.ContentSequence.append(warning_item)

        disclaimer_item = self._create_text_content(
            "CONTAINS",
            self._create_code("121131", "DCM", "Отказ от ответственности"),
            self.disclaimer
        )
        root_container.ContentSequence.append(disclaimer_item)

        # 8. Техническая информация
        tech_container = Dataset()
        tech_container.RelationshipType = "CONTAINS"
        tech_container.ValueType = "CONTAINER"
        tech_container.ConceptNameCodeSequence = [self._create_code("113876", "DCM", "Технические данные")]
        tech_container.ContinuityOfContent = "SEPARATE"
        tech_container.ContentSequence = []

        # Производитель и версия
        system_item = self._create_text_content(
            "CONTAINS",
            self._create_code("113878", "DCM", "Система"),
            f"{self.manufacturer} v{self.software_version}"
        )
        tech_container.ContentSequence.append(system_item)

        # Время анализа
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_item = self._create_text_content(
            "CONTAINS",
            self._create_code("111526", "DCM", "Время анализа"),
            analysis_time
        )
        tech_container.ContentSequence.append(time_item)

        root_container.ContentSequence.append(tech_container)

        # Добавляем корневой контейнер
        content_sequence.append(root_container)
        ds.ContentSequence = content_sequence

        return ds

    def _translate_status(self, status):
        """Переводит статус анализа на русский"""
        translations = {
            'normal': 'Норма',
            'atelectasis_only': 'Только ателектаз',
            'other_pathologies': 'Другие патологии'
        }
        return translations.get(status, status)

    def _translate_location(self, location):
        """Переводит локализацию на русский язык"""
        if not location or not isinstance(location, str):
            return location

        translations = {
            # Основные части легких
            'upper lobe': 'верхняя доля',
            'middle lobe': 'средняя доля',
            'lower lobe': 'нижняя доля',
            'lingula': 'язычок легкого',

            # Стороны
            'right lung': 'правое легкое',
            'left lung': 'левое легкое',

            # Зоны
            'upper zone': 'верхняя зона',
            'middle zone': 'средняя зона',
            'lower zone': 'нижняя зона',

            # Другие термины
            'diffuse': 'диффузное поражение',
            'focal': 'очаговое поражение'
        }

        # Применяем перевод для каждого термина
        result = location
        for eng, rus in translations.items():
            result = result.replace(eng, rus)

        return result.capitalize()

    def generate_sr_from_analysis(self, analysis_result, original_ds, output_path):
        """
        Генерирует DICOM SR из результатов анализа с соблюдением всех обязательных полей
        """
        try:
            print("=== ПРОВЕРКА ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ ===")

            # Проверяем обязательные поля
            missing_fields = self.validate_required_fields(analysis_result)
            if missing_fields:
                raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing_fields)}")

            print("✓ Все обязательные поля присутствуют")
            print(f"✓ Вероятность ателектаза: {analysis_result['atelectasis_probability']:.3f}")
            print(f"✓ Координаты bbox: {analysis_result['bbox']}")
            print(f"✓ Заключение: {analysis_result['conclusion'][:100]}...")

            # Создаем базовый SR dataset
            sr_ds = self.create_basic_sr_dataset(original_ds, analysis_result)

            # Добавляем содержимое с обязательными полями
            sr_ds = self.add_content_sequence(sr_ds, analysis_result, original_ds)

            # Сохраняем
            sr_ds.save_as(output_path)
            print(f"✓ DICOM SR успешно создан: {output_path}")

            # Выводим сводку по обязательным полям
            print("\n=== СВОДКА ОБЯЗАТЕЛЬНЫХ ПОЛЕЙ В DICOM SR ===")
            print(f"• Modality: {getattr(original_ds, 'Modality', 'DX')}")
            print(f"• ObserverType: {sr_ds.ObserverType}")
            print(f"• Findings: {analysis_result['conclusion'][:80]}...")
            print(f"• Probability: {analysis_result['atelectasis_probability']:.6f}")
            if analysis_result['bbox']:
                x_min, y_min, x_max, y_max = analysis_result['bbox']
                print(f"• BoundingBox: xmin={x_min}, ymin={y_min}, xmax={x_max}, ymax={y_max}")
            print(f"• Warning: Заключение сгенерировано ИИ. Требуется подтверждение врача.")
            print("==========================================\n")

            return True

        except Exception as e:
            print(f"❌ Ошибка при создании DICOM SR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


class AtelectasisPipeline:
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir

        # Инициализация компонентов
        self.dicom_processor = DicomProcessor()
        self.ai_detector = AtelectasisDetector(model_path)
        self.sr_generator = DicomSRGenerator()

        # Создание директорий
        self.json_dir = os.path.join(output_dir, "json_reports")
        self.sr_dir = os.path.join(output_dir, "dicom_sr")

        for dir_path in [self.json_dir, self.sr_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def process_dicom(self, dicom_path):
        """Полный пайплайн обработки DICOM"""
        print(f"[1/4] Обработка файла: {dicom_path}")

        # Check if DICOM file exists
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

        # Test DICOM file validity
        try:
            test_ds = pydicom.dcmread(dicom_path)
            print(f"[DEBUG] DICOM file is valid. Modality: {getattr(test_ds, 'Modality', 'Unknown')}")
            print(
                f"[DEBUG] Image size: {getattr(test_ds, 'Rows', 'Unknown')} x {getattr(test_ds, 'Columns', 'Unknown')}")
        except Exception as e:
            print(f"[ERROR] Invalid DICOM file: {e}")
            raise

        # Предобработка DICOM
        print("[2/4] Предобработка и анонимизация...")
        try:
            img_rgb, metadata, original_ds = self.dicom_processor.preprocess_dicom(dicom_path)
            print(f"[DEBUG] Preprocessed image shape: {img_rgb.shape}")
            print(f"[DEBUG] Metadata keys: {list(metadata.keys())}")
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            raise

        # Анализ ИИ
        print("[3/4] Анализ изображения нейросетью...")
        try:
            original_size = (original_ds.Columns, original_ds.Rows) if hasattr(original_ds, 'Columns') else (1024, 1024)
            print(f"[DEBUG] Original size: {original_size}")
            analysis_result = self.ai_detector.analyze_image(img_rgb, original_size)
            print(f"[DEBUG] Analysis result status: {analysis_result.get('status', 'Unknown')}")
        except Exception as e:
            print(f"[ERROR] AI analysis failed: {e}")
            raise

        # Добавляем метаданные и приводим к нужному формату полей
        analysis_result["metadata"] = metadata
        analysis_result["processing_timestamp"] = datetime.now().isoformat()

        # Ensure fields match E_Dicom_generation.py expectations
        if "status" not in analysis_result:
            analysis_result["status"] = "unknown"
        if "conclusion" not in analysis_result:
            analysis_result["conclusion"] = "No conclusion"
        if "bbox" not in analysis_result:
            analysis_result["bbox"] = []

        # Сохранение JSON
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        json_path = os.path.join(self.json_dir, f"{base_name}_report.json")

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=4)
            print(f"[DEBUG] JSON report saved: {json_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON report: {e}")

        # Создание DICOM SR using the new generator
        print("[4/4] Генерация DICOM SR...")
        sr_path = os.path.join(self.sr_dir, f"{base_name}_sr.dcm")

        try:
            success = self.sr_generator.generate_sr_from_analysis(analysis_result, original_ds, sr_path)
            if success:
                print("[✓] Обработка завершена успешно!")
                return {
                    "status": "success",
                    "json_report": json_path,
                    "dicom_sr": sr_path,
                    "results": analysis_result
                }
        except Exception as e:
            print(f"[!] Ошибка при создании DICOM SR: {e}")
            import traceback
            traceback.print_exc()

        return {
            "status": "partial_success",
            "json_report": json_path,
            "results": analysis_result
        }

    def process_multiple_dicoms(self, dicom_paths):
        """Обработка нескольких DICOM файлов"""
        results = []
        for dicom_path in dicom_paths:
            try:
                result = self.process_dicom(dicom_path)
                results.append(result)
            except Exception as e:
                print(f"Ошибка при обработке {dicom_path}: {e}")
                import traceback
                traceback.print_exc()
                results.append({"status": "error", "file": dicom_path, "error": str(e)})
        return results

    def create_test_dicom(self, png_path, output_dcm_path):
        """Создает тестовый DICOM из PNG"""
        return self.dicom_processor.create_dicom_from_png(png_path, output_dcm_path)


def main():
    # Настройки
    model_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\best_deit_scm_model_2.pth"
    output_dir = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\results"
    dicom_path = r"dicom.dcm"

    print("=== PIPELINE DEBUG ===")
    print(f"Model path exists: {os.path.exists(model_path)}")
    print(f"Output dir: {output_dir}")
    print(f"DICOM path exists: {os.path.exists(dicom_path)}")

    if not os.path.exists(dicom_path):
        print("ERROR: DICOM file doesn't exist. Run dicom_from_png.py first!")
        return

    try:
        # Создание и запуск пайплайна
        pipeline = AtelectasisPipeline(model_path, output_dir)
        result = pipeline.process_dicom(dicom_path)

        # Вывод результатов
        if result["status"] == "success":
            print("\n=== РЕЗУЛЬТАТЫ ===")
            print(f"Статус: {result['results']['status']}")
            print(f"Вероятность ателектаза: {result['results']['atelectasis_probability']:.2%}")
            if result['results']['bbox']:
                print(f"Локализация: {result['results']['bbox']}")
            print(f"Заключение: {result['results']['conclusion']}")
            print(f"\nJSON отчет: {result['json_report']}")
            if 'dicom_sr' in result:
                print(f"DICOM SR: {result['dicom_sr']}")
        else:
            print(f"\n=== PARTIAL SUCCESS ===")
            print(f"Status: {result['status']}")
            if 'results' in result:
                print(f"JSON report: {result['json_report']}")

    except Exception as e:
        print(f"PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()