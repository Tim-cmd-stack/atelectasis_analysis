# Example usage of the DicomProcessor class to convert PNG to DICOM
from sam_proekt.dicom_processor import DicomProcessor
import pydicom
import os

# Create an instance of DicomProcessor
processor = DicomProcessor(log_dir="logs")

# Convert PNG to DICOM
try:
    # Define paths
    png_path = r"C:\Users\CYBER ARTEL\.cache\kagglehub\datasets\nih-chest-xrays\data\nih_custom_dataset\new_classes\Atelectasis\00001101_012.png"
    output_dcm_path = r"C:\Users\CYBER ARTEL\PycharmProjects\atelectasis_classification_and_detection\sam_proekt\dicom4.dcm"

    # Check if input PNG file exists
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"Input PNG file not found: {png_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_dcm_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Converting PNG to DICOM...")
    print(f"Input: {png_path}")
    print(f"Output: {output_dcm_path}")

    # Method 1: With patient information (default)
    dicom_dataset = processor.create_dicom_from_png(
        png_path=png_path,
        output_dcm_path=output_dcm_path,
        add_patient_info=True  # This is default
    )

    # Verify the file was created
    if os.path.exists(output_dcm_path):
        print(f"Successfully created DICOM file: {output_dcm_path}")

        # Read and inspect the created DICOM
        ds = pydicom.dcmread(output_dcm_path)
        print(f"Patient Name: {ds.PatientName if hasattr(ds, 'PatientName') else 'Not set'}")
        print(f"Modality: {ds.Modality}")
        print(f"Image size: {ds.Rows} x {ds.Columns}")
        print(f"Study Description: {ds.StudyDescription}")
    else:
        print("Error: DICOM file was not created successfully")

except FileNotFoundError as e:
    print(f"File not found error: {e}")
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()  # This will show the full error traceback