#include <string>
#include "ultralytics/YOLO.h"

int main() {
    // Load a model
    // YOLO model("yolo11n-seg.pt");

    // Create YOLO instance
    YOLO model;

    // Training parameters
    TrainingConfig config;
    config.data = "configmodel.yaml";
    config.epochs = 1;
    config.imgsz = 640;

    // Train the model
    Results results = model.train(config);

    // Load trained model
    // YOLO trainedModel("best.pt");

    // Inference
    // Results inferenceResults = trainedModel.predict("D:/Desktop/Adas_test/ds_segmentation/images/img6.jpg");

    return 0;
}
