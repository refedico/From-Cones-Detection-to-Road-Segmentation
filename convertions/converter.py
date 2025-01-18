import os
import shutil
import random

def split_and_rename_dataset(dataset_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Splits a dataset with multiple subfolders into training and validation sets with unique filenames.
    
    Parameters:
        dataset_dir (str): Path to the dataset directory containing subfolders, each with 'images' and 'labels' subdirectories.
        output_dir (str): Path to output directory where 'train' and 'val' folders will be created.
        val_ratio (float): Fraction of data to be used for validation (e.g., 0.2 for 20%).
        seed (int): Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Define paths for output directories
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')
    
    # Create output directories if they don't exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Collect all images and labels from each subfolder
    all_data = []  # To store tuples of (image_path, label_path, unique_prefix)
    subfolders = [f.path for f in os.scandir(dataset_dir) if f.is_dir()]

    for subfolder in subfolders:
        images_path = os.path.join(subfolder, 'images')
        labels_path = os.path.join(subfolder, 'labels')
        
        if os.path.exists(images_path) and os.path.exists(labels_path):
            images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            prefix = os.path.basename(subfolder)  # Use subfolder name as prefix
            for image in images:
                image_path = os.path.join(images_path, image)
                label_path = os.path.join(labels_path, os.path.splitext(image)[0] + '.txt')
                
                # Add to list if both image and label exist
                if os.path.exists(label_path):
                    all_data.append((image_path, label_path, prefix))
    
    # Shuffle the data
    random.shuffle(all_data)

    # Split data into train and validation sets
    val_size = int(len(all_data) * val_ratio)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    # Function to move and rename files
    def move_and_rename_files(data_list, image_target_dir, label_target_dir):
        for image_path, label_path, prefix in data_list:
            # Create a new unique filename using the prefix and original filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            unique_image_name = f"{prefix}_{base_name}.png"
            unique_label_name = f"{prefix}_{base_name}.txt"
            
            # Copy image and label with new unique names
            shutil.copy(image_path, os.path.join(image_target_dir, unique_image_name))
            shutil.copy(label_path, os.path.join(label_target_dir, unique_label_name))

    # Move and rename image and label files to train/val directories
    move_and_rename_files(train_data, train_images_dir, train_labels_dir)
    move_and_rename_files(val_data, val_images_dir, val_labels_dir)

    print(f"Dataset split and renaming complete!")
    print(f"Training images: {len(train_data)}, Validation images: {len(val_data)}")

# Example usage
dataset_dir = 'convert'  # Path to the dataset with multiple subfolders
output_dir = 'trainable'  # Path to the output directory for train/val splits
split_and_rename_dataset(dataset_dir, output_dir, val_ratio=0.2, seed=42)
