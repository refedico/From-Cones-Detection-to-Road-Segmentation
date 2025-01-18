#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <cstring>

namespace fs = std::filesystem;

class STOY {
private:
    std::string source;
    std::string dest;
    bool checker;
    std::vector<std::string> source_list;
    std::map<std::string, int> class_to_ind;

public:
    STOY(const std::string& source_dir, const std::string& dest_dir, 
         const std::string& class_names, bool check) 
        : source(source_dir), dest(dest_dir), checker(check) {
        
        if (checker) {
            if (fs::exists("checking")) {
                fs::remove_all("checking");
            }
            fs::create_directory("checking");
        }

        // Read class names
        std::ifstream cls_file(class_names);
        std::string line;
        int index = 0;
        while (std::getline(cls_file, line)) {
            if (!line.empty() && line[line.length()-1] == '\n') {
                line.erase(line.length()-1);
            }
            class_to_ind[line] = index++;
        }
    }

    void find_source_list() {
        for (const auto& entry : fs::directory_iterator(source)) {
            if (fs::is_directory(entry)) {
                std::string dir_path = entry.path().string() + "/ann";
                if (fs::exists(dir_path)) {
                    source_list.push_back(dir_path);
                } else {
                    std::cout << "\nCan not find '" << dir_path << "' directory.\n" << std::endl;
                }
            }
        }
    }

    std::string conv2yolo(const nlohmann::json& json_obj, const std::string& txt_name, 
                         const std::string& label_path) {
        std::string lab_p = label_path + "/" + txt_name.substr(0, txt_name.find(".json"));
        std::string txt_path = lab_p;
        txt_path.replace(txt_path.find(".jpg"), 4, ".txt");
        txt_path.replace(txt_path.find(".png"), 4, ".txt");
        
        std::ofstream txt_file(txt_path);

        int w = json_obj["size"]["width"];
        int h = json_obj["size"]["height"];

        if (json_obj["objects"].empty()) {
            std::cout << "Error: " << txt_path << " is empty\n";
            txt_file.close();
            return lab_p;
        }

        cv::Mat img;
        if (checker) {
            std::string img_file = lab_p;
            img_file.replace(img_file.find(dest), dest.length(), source);
            img_file.replace(img_file.find("labels"), 6, "img");
            img = cv::imread(img_file);
        }

        for (const auto& object : json_obj["objects"]) {
            int cls = class_to_ind[object["classTitle"]];
            
            std::vector<int> points = object["points"]["exterior"];
            float x1 = points[0], y1 = points[1];
            float x2 = points[2], y2 = points[3];
            
            float x_cnt = ((x1 + x2) / 2.0f) / w;
            float y_cnt = ((y1 + y2) / 2.0f) / h;
            float width = std::abs((x2 - x1) / w);
            float height = std::abs((y2 - y1) / h);

            width = (width < 0) ? 0 : width;
            height = (height < 0) ? 0 : height;
            width = (width > w) ? w : width;
            height = (height > h) ? h : height;

            txt_file << cls << " " << std::fixed << std::setprecision(5) 
                    << x_cnt << " " << y_cnt << " " << width << " " << height << "\n";

            if (checker) {
                plot_one_box_ko({x1, y1, x2, y2}, img, std::to_string(cls));
            }
        }

        if (checker) {
            fs::path check_path = "checking/" + fs::path(lab_p).parent_path().filename().string();
            cv::imwrite(check_path.string() + "/" + fs::path(lab_p).filename().string(), img);
        }

        txt_file.close();
        return lab_p;
    }

    void plot_one_box_ko(const std::vector<float>& x, cv::Mat& img, 
                        const std::string& label = "", 
                        const cv::Scalar& color = cv::Scalar()) {
        int tl = std::round(0.002 * (img.rows + img.cols) / 2) + 1;
        
        cv::Scalar draw_color = color;
        if (color == cv::Scalar()) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 255);
            draw_color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        }

        cv::Point c1(static_cast<int>(x[0]), static_cast<int>(x[1]));
        cv::Point c2(static_cast<int>(x[2]), static_cast<int>(x[3]));
        cv::rectangle(img, c1, c2, draw_color, tl);

        if (!label.empty()) {
            int tf = std::max(tl - 1, 1);
            int baseline = 0;
            cv::Size t_size = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX_SMALL, 
                                            tl / 3.0, tf, &baseline);
            
            cv::putText(img, label, cv::Point(c1.x, c1.y - 2), 
                       cv::FONT_HERSHEY_COMPLEX_SMALL, tl / 3.0, 
                       cv::Scalar(225, 255, 255), tf, cv::LINE_AA);
        }
    }

    void run() {
        find_source_list();

        if (fs::exists(dest)) {
            fs::remove_all(dest);
        }
        fs::create_directory(dest);

        std::cout << "\n'" << dest << "' folder is created." << std::endl;

        std::ofstream all_path(dest + "/train.txt");

        for (const auto& dir : source_list) {
            fs::path dir_path = dest / fs::path(dir).parent_path().filename();
            fs::path label_path = dir_path / "labels";
            fs::path images_path = dir_path / "images";

            fs::create_directory(dir_path);
            fs::create_directory(label_path);
            fs::create_directory(images_path);

            if (checker) {
                fs::create_directory("checking/" + dir_path.filename().string());
                // Copy images (platform specific command)
                std::string cmd = "cp " + std::string(dir).replace(
                    dir.find("ann"), 3, "img") + "/*.* " + images_path.string();
                system(cmd.c_str());
            }

            for (const auto& entry : fs::directory_iterator(dir)) {
                std::ifstream json_file(entry.path());
                nlohmann::json source_json;
                json_file >> source_json;
                
                std::string lab_p = conv2yolo(source_json, 
                                            entry.path().filename().string(), 
                                            label_path.string());
                
                all_path << lab_p.replace(lab_p.find("labels"), 6, "images") << '\n';
            }
        }

        all_path.close();

        // Set permissions (platform specific command)
        system(("chmod 777 -R " + dest).c_str());
        if (checker) {
            system("chmod 777 -R checking");
        }
        std::cout << "\nAnnotation convert was done successfully.\n" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::string source_dir;
    std::string dest_dir = "convert";
    std::string class_names = "class.names";
    bool checker = false;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--source_dir" && i + 1 < argc) {
            source_dir = argv[++i];
        } else if (std::string(argv[i]) == "--dest_dir" && i + 1 < argc) {
            dest_dir = argv[++i];
        } else if (std::string(argv[i]) == "--name" && i + 1 < argc) {
            class_names = argv[++i];
        } else if (std::string(argv[i]) == "--checker") {
            checker = true;
        }
    }

    if (source_dir.empty()) {
        std::cout << "\nError: Folder is not exists." << std::endl;
        std::cout << "Please check your source_dir.\n" << std::endl;
        return 1;
    }

    STOY stoy(source_dir, dest_dir, class_names, checker);
    stoy.run();

    return 0;
}
