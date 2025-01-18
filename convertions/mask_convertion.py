#from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
import cv2
from pathlib import Path
import numpy as np

def convert_segment_masks_to_yolo_seg(masks_dir, output_dir, classes):
    """
    Converts a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation format.
    The converted masks are saved in the specified output directory.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        classes (int): Total classes in the dataset i.e. for COCO classes=80
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix == ".png" or mask_path.suffix == ".jpg":
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            img_height, img_width = mask.shape  # Get image dimensions
            print(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # Skip background
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    print(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # Find contours

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        yolo_format = [class_index]
                        for point in contour:
                            # Normalize the coordinates
                            yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # Save Ultralytics YOLO format data to file
            output_path = Path(output_dir) / f"{mask_path.stem}.txt"
            with open(output_path, "w") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            print(f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}")

def main():
    mask_path = "D:/Desktop/Adas_test/ds_segmentation/masks/"
    txt_path = "D:/Desktop/Adas_test/ds_segmentation/txt_masks/"
    convert_segment_masks_to_yolo_seg(mask_path, txt_path, classes=2)

main()
