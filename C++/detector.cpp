#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>
#include <numeric>
#include <tuple>
#include <map>

// Constants
const int YELLOW_INDEX = 4;
const int BLUE_INDEX = 0;
const int CLASS_NUM = 5;

// Function to generate a binary mask
cv::Mat generate_binary_mask(const cv::Mat& img, std::vector<cv::Point>& yellow_points, std::vector<cv::Point>& blue_points, const std::string& mask_path) {
    if (blue_points.empty()) {
        blue_points.insert(blue_points.begin(), cv::Point(0, img.rows));
    } else if (blue_points.size() == 1) {
        blue_points.insert(blue_points.begin(), cv::Point(0, img.rows));
    } else {
        int x1 = blue_points[0].x, y1 = blue_points[0].y;
        int x2 = blue_points[1].x, y2 = blue_points[1].y;
        double m = static_cast<double>(y2 - y1) / (x2 - x1);
        double q = y1 - m * x1;
        int x = 0;
        int y = static_cast<int>(m * x + q);
        if (m > 0) {
            blue_points.insert(blue_points.begin(), cv::Point(0, img.rows));
        } else {
            blue_points.insert(blue_points.begin(), cv::Point(x, y));
            blue_points.insert(blue_points.begin(), cv::Point(0, img.rows));
        }
    }
    if (yellow_points.empty()) {
        yellow_points.insert(yellow_points.begin(), cv::Point(img.cols, img.rows));
    } else if (yellow_points.size() == 1) {
        yellow_points.insert(yellow_points.begin(), cv::Point(img.cols, img.rows));
    } else {
        int x1 = yellow_points[0].x, y1 = yellow_points[0].y;
        int x2 = yellow_points[1].x, y2 = yellow_points[1].y;
        double m = static_cast<double>(y2 - y1) / (x2 - x1);
        double q = y1 - m * x1;
        int x = img.cols;
        int y = static_cast<int>(m * x + q);
        if (m < 0) {
            yellow_points.insert(yellow_points.begin(), cv::Point(img.cols, img.rows));
        } else {
            yellow_points.insert(yellow_points.begin(), cv::Point(x, y));
            yellow_points.insert(yellow_points.begin(), cv::Point(img.cols, img.rows));
        }
    }
    cv::Mat mask_f = cv::Mat::zeros(img.size(), CV_8UC1);
    std::vector<cv::Point> points;
    points.insert(points.end(), yellow_points.begin(), yellow_points.end());
    points.insert(points.end(), blue_points.rbegin(), blue_points.rend());
    cv::fillPoly(mask_f, std::vector<std::vector<cv::Point>>{points}, cv::Scalar(255));
    cv::imwrite(mask_path, mask_f);
    return mask_f;
}

// Function to order images
std::vector<std::string> order_img(const std::string& img_dir) {
    std::vector<std::string> imgs;
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        imgs.push_back(entry.path().filename().string());
    }
    std::sort(imgs.begin(), imgs.end());
    return imgs;
}

// Function to find the box with the maximum height
int max_box(const std::vector<cv::Rect>& boxes) {
    int max_height = 0;
    int max_box = 0;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int height = boxes[i].height;
        if (height > max_height) {
            max_height = height;
            max_box = i;
        }
    }
    return max_box;
}

// Function to find the lowest point
int lowest_point(const std::vector<cv::Point>& points) {
    int lowest = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].y > points[lowest].y) {
            lowest = i;
        }
    }
    return lowest;
}

// Function to check if two line segments intersect
bool doesIntersect(const cv::Point& p1, const cv::Point& q1, const cv::Point& p2, const cv::Point& q2) {
    auto onSegment = [](const cv::Point& p, const cv::Point& q, const cv::Point& r) {
        return (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
                q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y));
    };

    auto orientation = [](const cv::Point& p, const cv::Point& q, const cv::Point& r) {
        int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if (val > 0) return 1;
        if (val < 0) return 2;
        return 0;
    };

    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;

    return false;
}

// Function to calculate the angle between three points
double getAngle(const cv::Point& a, const cv::Point& b, const cv::Point& c) {
    double ang = std::atan2(c.y - b.y, c.x - b.x) - std::atan2(a.y - b.y, a.x - b.x);
    ang = ang * 180.0 / CV_PI;
    return ang < 0 ? ang + 360 : ang;
}

// Function to list graphs
std::tuple<std::vector<std::vector<std::tuple<int, int, double>>>, std::vector<std::vector<cv::Rect>>, std::vector<std::vector<cv::Point>>>
list_graphs(const std::vector<cv::Rect>& boxes, const std::vector<cv::Point>& points, const std::vector<int>& class_ids) {
    std::vector<std::vector<std::tuple<int, int, double>>> graphs;
    std::vector<std::vector<cv::Rect>> boxes_classified(CLASS_NUM);
    std::vector<std::vector<cv::Point>> points_classified(CLASS_NUM);

    for (int i = 0; i < CLASS_NUM; ++i) {
        std::vector<cv::Rect> boxes_class;
        std::vector<cv::Point> mid_points_class;
        for (size_t j = 0; j < boxes.size(); ++j) {
            if (class_ids[j] == i) {
                boxes_class.push_back(boxes[j]);
                mid_points_class.push_back(points[j]);
            }
        }
        graphs.push_back(graph_generator(boxes_class, mid_points_class));
        boxes_classified[i] = boxes_class;
        points_classified[i] = mid_points_class;
    }
    return {graphs, boxes_classified, points_classified};
}

// Function to generate a graph
std::vector<std::tuple<int, int, double>> graph_generator(const std::vector<cv::Rect>& boxes, const std::vector<cv::Point>& points) {
    std::vector<std::tuple<int, int, double>> graph;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < points.size(); ++j) {
            if (i != j) {
                double height1 = boxes[i].height;
                double height2 = boxes[j].height;
                double distance = std::sqrt(std::pow(points[i].x - points[j].x, 2) + std::pow(points[i].y - points[j].y, 3)) * (1 / height1 + 1 / height2);
                graph.emplace_back(i, j, distance);
            }
        }
    }
    std::sort(graph.begin(), graph.end(), [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
    return graph;
}

// Function to perform inference
std::pair<cv::Mat, cv::Mat> inference(torch::jit::script::Module& model, const std::vector<std::string>& imgs, int i) {
    // Inference
    auto results = model.forward({torch::from_blob(imgs.data(), {static_cast<int64_t>(imgs.size())}, torch::kInt32)}).toTuple();

    // Define class colors
    std::map<int, cv::Scalar> CLASS_COLORS = {
        {0, cv::Scalar(255, 0, 0)},    // Red
        {1, cv::Scalar(0, 255, 0)},    // Green
        {2, cv::Scalar(0, 0, 255)},    // Blue
        {3, cv::Scalar(255, 255, 0)},  // Yellow
        {4, cv::Scalar(255, 165, 0)},  // Orange
        {5, cv::Scalar(128, 0, 128)},  // Purple
        {6, cv::Scalar(0, 255, 255)},  // Cyan
        {7, cv::Scalar(255, 192, 203)},// Pink
        {8, cv::Scalar(128, 128, 0)}   // Olive
    };

    // Function to draw colored boxes
    auto draw_colored_boxes = [&](cv::Mat& img, const std::vector<cv::Rect>& boxes, const std::vector<int>& class_ids) {
        for (size_t i = 0; i < boxes.size(); ++i) {
            cv::rectangle(img, boxes[i], CLASS_COLORS[class_ids[i]], 3);
        }
    };

    // Function to draw colored points
    auto draw_colored_points = [&](cv::Mat& img, const std::vector<cv::Point>& mid_points, const std::vector<int>& class_ids) {
        for (size_t i = 0; i < mid_points.size(); ++i) {
            cv::circle(img, mid_points[i], 5, CLASS_COLORS[class_ids[i]], cv::FILLED);
        }
    };

    // Function to check if a point is an outlier
    auto isoutlier = [&](int x1, int x2, int x3, int y1, int y2, int y3) {
        double angle = getAngle(cv::Point(x1, y1), cv::Point(x2, y2), cv::Point(x3, y3)) - 180;
        angle = std::abs(angle) > 160;
        double distance = std::sqrt(std::pow(x2 - x3, 2) + std::pow(y2 - y3, 2));
        distance = distance > 3 * std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
        return angle || distance;
    };

    // Function to draw colored lines
    auto draw_colored_lines = [&](cv::Mat& img, const std::vector<std::vector<cv::Point>>& points_classified, const std::vector<std::vector<cv::Rect>>& boxes_classified, const std::vector<std::vector<std::tuple<int, int, double>>>& graphs) {
        std::vector<cv::Point> yellow_points;
        std::vector<cv::Point> blue_points;
        for (int class_id = 0; class_id < CLASS_NUM; ++class_id) {
            if (boxes_classified[class_id].empty()) continue;
            if (class_id != YELLOW_INDEX && class_id != BLUE_INDEX) continue;
            const auto& graph = graphs[class_id];
            const auto& boxes = boxes_classified[class_id];
            const auto& mid_points = points_classified[class_id];
            int starting_index = lowest_point(mid_points);
            std::vector<int> connected = {starting_index};
            int current_index = starting_index;
            int last_index = starting_index;
            bool ended = false;
            if (class_id == YELLOW_INDEX) {
                yellow_points.push_back(mid_points[starting_index]);
            } else if (class_id == BLUE_INDEX) {
                blue_points.push_back(mid_points[starting_index]);
            }
            while (connected.size() < boxes.size() && !ended) {
                for (const auto& [i, j, distance] : graph) {
                    if (i == current_index && std::find(connected.begin(), connected.end(), j) == connected.end()) {
                        int x2 = mid_points[i].x, y2 = mid_points[i].y;
                        int x3 = mid_points[j].x, y3 = mid_points[j].y;
                        int x1 = mid_points[last_index].x, y1 = mid_points[last_index].y;
                        if (i == last_index || !isoutlier(x1, x2, x3, y1, y2, y3)) {
                            if (class_id == YELLOW_INDEX) {
                                yellow_points.push_back(mid_points[j]);
                            } else if (class_id == BLUE_INDEX) {
                                blue_points.push_back(mid_points[j]);
                            }
                        } else {
                            ended = true;
                            break;
                        }
                        connected.push_back(j);
                        current_index = j;
                        last_index = i;
                        break;
                    }
                }
            }
        }
        if (yellow_points.size() > 1 && blue_points.size() > 1) {
            for (size_t i = 1; i < yellow_points.size(); ++i) {
                int x1 = yellow_points[i - 1].x, y1 = yellow_points[i - 1].y;
                int x2 = yellow_points[i].x, y2 = yellow_points[i].y;
                for (size_t j = 1; j < blue_points.size(); ++j) {
                    int x3 = blue_points[j - 1].x, y3 = blue_points[j - 1].y;
                    int x4 = blue_points[j].x, y4 = blue_points[j].y;
                    if (doesIntersect(cv::Point(x1, y1), cv::Point(x2, y2), cv::Point(x3, y3), cv::Point(x4, y4))) {
                        double distance1 = std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 3));
                        double distance2 = std::sqrt(std::pow(x3 - x4, 2) + std::pow(y3 - y4, 3));
                        if (distance1 > distance2) {
                            yellow_points.resize(i);
                        } else {
                            blue_points.resize(j);
                        }
                    }
                }
            }
        }
        if (yellow_points.size() > 1 && blue_points.size() > 1) {
            int x1 = img.cols, y1 = img.rows;
            int x2 = yellow_points[0].x, y2 = yellow_points[0].y;
            for (size_t j = 1; j < blue_points.size(); ++j) {
                int x3 = blue_points[j - 1].x, y3 = blue_points[j - 1].y;
                int x4 = blue_points[j].x, y4 = blue_points[j].y;
                if (doesIntersect(cv::Point(x1, y1), cv::Point(x2, y2), cv::Point(x3, y3), cv::Point(x4, y4))) {
                    yellow_points.clear();
                    break;
                }
            }
        }
        if (yellow_points.size() > 1 && blue_points.size() > 1) {
            int x1 = 0, y1 = img.rows;
            int x2 = blue_points[0].x, y2 = blue_points[0].y;
            for (size_t i = 1; i < yellow_points.size(); ++i) {
                int x3 = yellow_points[i - 1].x, y3 = yellow_points[i - 1].y;
                int x4 = yellow_points[i].x, y4 = yellow_points[i].y;
                if (doesIntersect(cv::Point(x1, y1), cv::Point(x2, y2), cv::Point(x3, y3), cv::Point(x4, y4))) {
                    blue_points.clear();
                    break;
                }
            }
        }
        return std::make_pair(img, std::make_pair(blue_points, yellow_points));
    };

    // Function to calculate Intersection over Union (IoU)
    auto iou = [](const cv::Rect& box1, const cv::Rect& box2) {
        int x1 = std::max(box1.x, box2.x);
        int y1 = std::max(box1.y, box2.y);
        int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
        int intersection_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
        int area_box1 = box1.width * box1.height;
        int area_box2 = box2.width * box2.height;
        int union_area = area_box1 + area_box2 - intersection_area;
        return static_cast<double>(intersection_area) / union_area;
    };

    // Perform inference
    results = model.forward({torch::from_blob(imgs.data(), {static_cast<int64_t>(imgs.size())}, torch::kInt32)}).toTuple();

    // Get bounding boxes and classes
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    // Assuming results[0] contains the bounding boxes and class IDs
    // This part needs to be adapted based on the actual output format of the model
    // For example:
    // boxes = results[0].toTensor().cpu().data().numpy();
    // class_ids = results[1].toTensor().cpu().data().numpy();
    // confidences = results[2].toTensor().cpu().data().numpy();

    // Remove overlapping boxes
    for (size_t i = 0; i < boxes.size(); ++i) {
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (iou(boxes[i], boxes[j]) > 0.6) {
                if (confidences[i] > confidences[j]) {
                    boxes[j] = cv::Rect();
                } else {
                    boxes[i] = cv::Rect();
                }
            }
        }
    }
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const cv::Rect& box) { return box.area() == 0; }), boxes.end());

    // Remove boxes with low confidence
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [&](const cv::Rect& box) { return confidences[&box - &boxes[0]] < 0.5; }), boxes.end());

    // Remove small boxes
    int max_height = 0;
    for (const auto& box : boxes) {
        max_height = std::max(max_height, box.height);
    }
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [&](const cv::Rect& box) { return box.height < 0.10 * max_height; }), boxes.end());

    // Find mid points
    std::vector<cv::Point> mid_points;
    for (const auto& box : boxes) {
        mid_points.emplace_back((box.x + box.x + box.width) / 2, box.y + box.height);
    }

    // Generate graphs
    auto start = std::chrono::high_resolution_clock::now();
    auto [graphs, boxes_classified, points_classified] = list_graphs(boxes, mid_points, class_ids);
    auto end1 = std::chrono::high_resolution_clock::now();

    cv::Mat img = cv::imread(imgs[0]);

    // Draw colored lines
    auto [img_with_colored_boxes, points] = draw_colored_lines(img, points_classified, boxes_classified, graphs);
    auto [blue_points, yellow_points] = points;

    // Generate binary mask
    std::string mask_path = "/home/root/ADAS/mask.jpg";
    cv::Mat mask = generate_binary_mask(img, yellow_points, blue_points, mask_path);

    // Blend mask with image
    cv::Mat img_sum;
    cv::addWeighted(img_with_colored_boxes, 0.8, cv::Mat(img.size(), CV_8UC3, cv::Scalar(255, 255, 255)), 0.2, 0, img_sum);

    // Rearrange colors
    cv::cvtColor(img_sum, img_sum, cv::COLOR_BGR2RGB);

    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time for graph generation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start).count() << " ms\n";
    std::cout << "Time for drawing lines: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count() << " ms\n";

    // Save result
    cv::imwrite("output.jpg", img_with_colored_boxes);

    // Display result
    cv::Mat img_cv = cv::imread("output.jpg");
    cv::resize(img_cv, img_cv, cv::Size(800, 600));
    cv::imshow("Result", img_cv);
    cv::resize(img_sum, img_sum, cv::Size(800, 600));
    cv::imshow("Result with mask", img_sum);
    cv::waitKey(0);

    return {img, mask};
}

// Main function
int main() {
    // Load model
    torch::jit::script::Module model = torch::jit::load("unipr-detect.pt");
    torch::jit::script::Module seg_model = torch::jit::load("yolo-segment.pt");

    // Image directory
    std::string img_dir = "/home/root/ADAS/Dataset/amz/img/";
    bool CROP = true;

    // Order images
    auto imgs = order_img(img_dir);

    // Crop images if needed
    if (CROP) {
        for (const auto& img_name : imgs) {
            if (img_name.ends_with(".jpg") || img_name.ends_with(".png")) {
                std::string img_path = img_dir + img_name;
                // Crop logic here
            }
        }
    }

    return 0;
}
