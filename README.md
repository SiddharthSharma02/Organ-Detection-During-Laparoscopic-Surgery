# Organ Detection using YOLOv8
This project uses YOLOv8 to detect organs in medical imagery. The model has been trained using a custom dataset and is capable of segmenting and classifying various organs with high accuracy. Then tensorflow lite is used for it's deployment.

# Table of Contents
Overview<br>
Dataset<br>
Usage<br>
Model Training<br>
Model Evaluation<br>
Results<br>

# Overview
This repository provides the implementation of YOLOv8 for detecting and segmenting organs from medical images, including:<br>

Liver<br>
Gallbladder<br>
Hepatic Vein<br>
Abdominal Wall<br>
Gastrointestinal Tract<br>
Cystic Duct<br>
And others...<br>
The goal is to automate the detection and classification of organs in medical images using computer vision techniques.<br>

Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/organ-detection-yolov8.git
cd organ-detection-yolov8
Install Required Packages: Install the necessary libraries by running the following command:

bash
Copy code
pip install -r requirements.txt
Dependencies:

ultralytics (for YOLOv8)
opencv-python (for image processing)
matplotlib (for visualization)
numpy (for numerical operations)
wandb (for experiment tracking)
Setup Google Drive (if using Colab): If you're using Google Colab, mount your Google Drive to access datasets:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Dataset
The dataset consists of medical images and corresponding masks for each organ. The dataset is structured as follows:

train_images: Images for training the model.
val_images: Images for validation.
test_images: Images for testing.
train_labels: Labels for training images.
val_labels: Labels for validation images.
test_labels: Labels for test images.
The dataset is organized under the following directories:

bash
Copy code
/output
    /images
        /train
        /validation
        /test
    /labels
        /train
        /validation
        /test
Please refer to the config.yaml file for the correct path settings for your dataset.

Usage
1. Inference on a Single Image:
To run the trained YOLO model for detecting organs on a single image, use the following command:

python
Copy code
from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')  # Load the pre-trained model
results = model('path_to_image.jpg')  # Inference on a single image

# Visualize results
results.show()
2. Inference from Camera Feed:
You can also run inference on live camera feed:

python
Copy code
import cv2
from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')

cap = cv2.VideoCapture(0)  # Open webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform inference
    results.show()  # Display the results

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Model Training
1. Training the YOLOv8 Model:
To train the model on your dataset, use the following script:

python
Copy code
from ultralytics import YOLO

# Load the YOLOv8 architecture and pre-trained weights
model = YOLO('yolov8m-seg.yaml').load('yolov8m-seg.pt')

# Train the model
model.train(
    data='path_to_dataset.yaml',  # Dataset config file
    epochs=50,  # Number of epochs
    batch=16,  # Batch size
    imgsz=640,  # Image size
    device=0  # Set to -1 for CPU, or 0 for GPU
)
2. Training Configuration (dataset.yaml):
Your dataset configuration file should be structured as follows:

yaml
Copy code
path: /content/drive/My Drive/output  # Root path to the dataset
train: images/train  # Training images folder
val: images/validation  # Validation images folder
test: images/test  # Test images folder

nc: 13  # Number of classes
names:
  - Black Background
  - Abdominal Wall
  - Liver
  - Gastrointestinal Tract
  - Fat
  - Grasper
  - Connective Tissue
  - Blood
  - Cystic Duct
  - L-hook Electrocautery
  - Gallbladder
  - Hepatic Vein
  - Liver Ligament
Model Evaluation
After training, evaluate the model using the following code:

python
Copy code
metrics, map50 = model.val(data='path_to_dataset.yaml', split='test')
print(f"mAP@50: {map50}")
The evaluation will give you the mAP@50 (mean average precision at IoU threshold 0.5), a key metric to assess the model's performance.

Results
Once the model is trained, you can evaluate its performance on the test dataset. The mAP@50 score is the primary metric, where higher values indicate better model performance.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Notes:
Make sure to adjust paths in the code according to your local setup.
Use GPU for faster training if possible, especially for large datasets.
This README.md serves as a guide for setting up, using, and training the YOLOv8 model for organ detection. Feel free to modify it according to your project specifics.
