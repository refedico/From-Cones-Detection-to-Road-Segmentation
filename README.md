# From Cones Detection to Road Segmentation

## Description

This project was developed for the [Autonomous Driving and Adas Technologies course](https://corsi.unipr.it/en/ugov/degreecourse/225616) at the University of [Parma](https://www.unipr.it/), in collaboration with my colleague [Leonardo Zanella](https://github.com/leokx6). The two objectives are the identification of cones along the road of Formula SAE competitions and the identification through segmentation of the road that can actually be driven. Initially, after the first detection, it was thought to use a deterministic algorithm to identify the lines and the center of the road to drive the car accordingly. It was subsequently used to generate ground truth for training the real-time semantic segmentation model that will run on the car.

## System Requirements

Make sure you have installed:

    - Python 3.8 or higher
    - pip (Python package manager)

## Installing Requirements

To run this project, install the Python dependencies listed in requirements.txt using the following command:

```bash
pip install -r requirements.txt
```

### Contents of "requirements.txt"

Here are the libraries required for the project:

```
torch
ultralytics
Pillow
opencv-python
numpy
```

## Project Structure

The project structure is organized as follows:
```
.  
├── convertions/             # Modules for handling conversions  
│   └── mask_convertion.py 
│   └── conv2yolo.py.py 
│   └── converter.py 
├── documentation/           # Documentation-related files  
│   └── doc.html              
├── models/                  # Directory for models  
│   └── yolo.pt             
├── README.md                 
├── configmodel.yaml         # Configuration file for the model  
├── detector.py              # MAIN Script for object detection (TO START) 
├── requirements.txt   
├── seg_model.py             # Training segmentation model  
```

## How to Run
1. Clone or download the project repository:

   ```bash
   git clone https://github.com/refedico/From-Cones-Detection-to-Road-Segmentation
   cd From-Cones-Detection-to-Road-Segmentation
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python detector.py
   ```

## Notes
- Ensure that your environment supports GPU (optional but recommended for using YOLO models).
- Verify that you have the necessary permissions to access files in the data/ directory and to write to the outputs/ directory.

## Support
If you encounter any issues, contact the project maintainer or open an issue on the GitHub repository.
