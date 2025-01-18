#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <string>

namespace fs = std::filesystem;

void convert_segment_masks_to_yolo_seg(const std::string& masks_dir, 
                                     const std::string& output_dir, 
                                     int classes) {
    /*
    Converts a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation format.
    The converted masks are saved in the specified output directory.

    Args:
        masks_dir: The path to the directory where all mask images (png, jpg) are stored.
        output_dir: The path to the directory where the converted YOLO segmentation masks will be stored.
        classes: Total classes in the dataset i.e. for COCO classes=80
    */
    
    std::map<int, int> pixel_to_class_mapping;
    for (int i = 0; i < classes; ++i) {
        pixel_to_class_mapping[i + 1] = i;
    }

    for (const auto& entry : fs::directory_iterator(masks_dir)) {
        std::string extension = entry.path().extension().string();
        if (extension != ".png" && extension != ".jpg") continue;

        // Read the mask image in grayscale
        cv::Mat mask = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        int img_height = mask.rows;
        int img_width = mask.cols;
        
        std::cout << "Processing " << entry.path().string() 
                 << " imgsz = " << img_height << " x " << img_width << std::endl;

        // Get unique pixel values
        std::vector<int> unique_values;
        for(int i = 0; i < img_height; i++) {
            for(int j = 0; j < img_width; j++) {
                int value = mask.at<uchar>(i,j);
                if(std::find(unique_values.begin(), unique_values.end(), value) == unique_values.end()) {
                    unique_values.push_back(value);
                }
            }
        }

        std::vector<std::vector<double>> yolo_format_data;

        for (int value : unique_values) {
            if (value == 0) continue; // Skip background

            auto it = pixel_to_class_mapping.find(value);
            if (it == pixel_to_class_mapping.end()) {
                std::cout << "Unknown class for pixel value " << value 
                         << " in file " << entry.path() << ", skipping." << std::endl;
                continue;
            }
            int class_index = it->second;

            // Create a binary mask for the current class
            cv::Mat binary_mask;
            cv::compare(mask, value, binary_mask, cv::CMP_EQ);
            binary_mask.convertTo(binary_mask, CV_8U);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                if (contour.size() >= 3) { // YOLO requires at least 3 points
                    std::vector<double> yolo_format;
                    yolo_format.push_back(class_index);
                    
                    for (const auto& point : contour) {
                        // Normalize the coordinates
                        yolo_format.push_back(round(point.x * 1000000.0 / img_width) / 1000000.0);
                        yolo_format.push_back(round(point.y * 1000000.0 / img_height) / 1000000.0);
                    }
                    yolo_format_data.push_back(yolo_format);
                }
            }
        }

        // Save Ultralytics YOLO format data to file
        fs::path output_path = fs::path(output_dir) / (entry.path().stem().string() + ".txt");
        std::ofstream file(output_path);
        
        for (const auto& item : yolo_format_data) {
            for (size_t i = 0; i < item.size(); ++i) {
                file << item[i];
                if (i < item.size() - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
        
        std::cout << "Processed and stored at " << output_path 
                 << " imgsz = " << img_height << " x " << img_width << std::endl;
    }
}

int main() {
    std::string mask_path = "D:/Desktop/Adas_test/ds_segmentation/masks/";
    std::string txt_path = "D:/Desktop/Adas_test/ds_segmentation/txt_masks/";
    convert_segment_masks_to_yolo_seg(mask_path, txt_path, 2);
    return 0;
}
