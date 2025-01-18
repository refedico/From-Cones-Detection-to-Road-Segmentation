#include <iostream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <tuple>
#include <fstream>

namespace fs = std::filesystem;

void split_and_rename_dataset(const std::string& dataset_dir, const std::string& output_dir, 
                            float val_ratio = 0.2, int seed = 42) {
    /*
    Splits a dataset with multiple subfolders into training and validation sets with unique filenames.
    
    Parameters:
        dataset_dir: Path to the dataset directory containing subfolders, each with 'images' and 'labels' subdirectories.
        output_dir: Path to output directory where 'train' and 'val' folders will be created.
        val_ratio: Fraction of data to be used for validation (e.g., 0.2 for 20%).
        seed: Random seed for reproducibility.
    */
    
    // Set random seed for reproducibility
    std::mt19937 rng(seed);
    
    // Define paths for output directories
    fs::path train_images_dir = fs::path(output_dir) / "images" / "train";
    fs::path val_images_dir = fs::path(output_dir) / "images" / "val";
    fs::path train_labels_dir = fs::path(output_dir) / "labels" / "train";
    fs::path val_labels_dir = fs::path(output_dir) / "labels" / "val";
    
    // Create output directories if they don't exist
    fs::create_directories(train_images_dir);
    fs::create_directories(val_images_dir);
    fs::create_directories(train_labels_dir);
    fs::create_directories(val_labels_dir);
    
    // Collect all images and labels from each subfolder
    std::vector<std::tuple<fs::path, fs::path, std::string>> all_data;
    
    for(const auto& entry : fs::directory_iterator(dataset_dir)) {
        if(fs::is_directory(entry)) {
            fs::path subfolder = entry.path();
            fs::path images_path = subfolder / "images";
            fs::path labels_path = subfolder / "labels";
            
            if(fs::exists(images_path) && fs::exists(labels_path)) {
                std::string prefix = subfolder.filename().string();
                
                for(const auto& image : fs::directory_iterator(images_path)) {
                    std::string ext = image.path().extension().string();
                    if(ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                        fs::path image_path = image.path();
                        fs::path label_path = labels_path / 
                            (image_path.stem().string() + ".txt");
                        
                        if(fs::exists(label_path)) {
                            all_data.emplace_back(image_path, label_path, prefix);
                        }
                    }
                }
            }
        }
    }
    
    // Shuffle the data
    std::shuffle(all_data.begin(), all_data.end(), rng);
    
    // Split data into train and validation sets
    size_t val_size = static_cast<size_t>(all_data.size() * val_ratio);
    std::vector<std::tuple<fs::path, fs::path, std::string>> val_data(
        all_data.begin(), all_data.begin() + val_size);
    std::vector<std::tuple<fs::path, fs::path, std::string>> train_data(
        all_data.begin() + val_size, all_data.end());
    
    // Lambda function to move and rename files
    auto move_and_rename_files = [](const auto& data_list, 
                                  const fs::path& image_target_dir,
                                  const fs::path& label_target_dir) {
        for(const auto& [image_path, label_path, prefix] : data_list) {
            // Create a new unique filename using the prefix and original filename
            std::string base_name = image_path.stem().string();
            std::string unique_image_name = prefix + "_" + base_name + ".png";
            std::string unique_label_name = prefix + "_" + base_name + ".txt";
            
            // Copy image and label with new unique names
            fs::copy_file(image_path, 
                         image_target_dir / unique_image_name,
                         fs::copy_options::overwrite_existing);
            fs::copy_file(label_path,
                         label_target_dir / unique_label_name,
                         fs::copy_options::overwrite_existing);
        }
    };
    
    // Move and rename image and label files to train/val directories
    move_and_rename_files(train_data, train_images_dir, train_labels_dir);
    move_and_rename_files(val_data, val_images_dir, val_labels_dir);
    
    std::cout << "Dataset split and renaming complete!" << std::endl;
    std::cout << "Training images: " << train_data.size() 
              << ", Validation images: " << val_data.size() << std::endl;
}

int main() {
    std::string dataset_dir = "convert";  // Path to the dataset with multiple subfolders
    std::string output_dir = "trainable";  // Path to the output directory for train/val splits
    split_and_rename_dataset(dataset_dir, output_dir, 0.2, 42);
    return 0;
}
