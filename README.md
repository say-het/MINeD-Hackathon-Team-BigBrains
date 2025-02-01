# Project Showcase: BigBrains Final Project

## Overview
This document showcases how we developed our project, which focuses on detecting wagons and identifying potential defects such as residue, small holes, and cracks using deep learning techniques. We utilized **Detectron2** for object detection and **ReportLab** for generating reports. The following sections outline the step-by-step process of our project, including dataset preparation, model training, and inference.

---

## Steps in Our Process

### Step 1: Detecting Wagons Using Intensity
We first detected wagons based on their intensity differences in the input video frames. This allowed us to separate individual wagons from the background and store them as separate images.

### Step 2: Storing Wagon Images
Each detected wagon was stored as an individual image file in the **wagon** folder with names like `wagon_1.jpg`, `wagon_2.jpg`, etc.

### Step 3: Detecting Residue, Small Holes, and Cracks
Using our trained **Mask R-CNN** model, we analyzed the wagon images to detect defects, including:
- **Residue**
- **Small Holes**
- **Cracks**

### Step 4: Saving Detected Wagon Images
The detected wagons, now with annotations highlighting the defects, were saved in the **detected_wagons** folder with filenames such as `detected_wagon_1.jpg`, `detected_wagon_2.jpg`, etc.

### Step 5: Generating Reports with Gemini LLM
We then used **Gemini LLM** to generate a detailed report based on the defect detection results. The report includes information about the defects, their locations, and severity.

---

## How We Trained Our Model

### Step 1: Using Detectron2
We used **Detectron2**, a powerful object detection framework from Facebook AI, for training our model. The dependencies installed included:
- **PyTorch** for deep learning computations
- **Detectron2** for object detection
- **OpenCV** for image processing
- **NumPy** for numerical computations
- **Pillow** for handling image formats
- **ReportLab** for report generation
- **Google Generative AI** for report automation

### Step 2: Dataset Preparation and Annotation
To train the model, we extracted individual wagon frames from videos and manually annotated them using **CVAT** (Computer Vision Annotation Tool). These labeled images were then compiled into a structured dataset for training.

### Step 3: Model Training
We built a **Mask R-CNN** model for defect detection, achieving:
- **Accuracy**: Nearly **95%**
- **Total Loss**: **0.1047**

---

## Running the Project

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Jupyter Notebook or Jupyter Lab**
- Required dependencies:
  ```bash
  pip install torch torchvision torchaudio
  pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html
  pip install numpy opencv-python pillow reportlab google-generativeai
  ```

### Running the Notebook
1. Open a terminal and navigate to the project directory:
   ```bash
   jupyter notebook BigBrains-Final-Project-JK-Lakshmi.ipynb
   ```
2. Run the cells sequentially to process images, perform inference, and generate reports.

### Expected Outputs
- Processed images with detected wagons and defect annotations.
- Generated PDF reports containing details about identified defects.

### Troubleshooting
- If you face module import errors, ensure all dependencies are correctly installed.
- Check GPU availability with:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
  If **False**, consider installing CUDA drivers or using a cloud service like **Google Colab**.

---

## Conclusion
This project successfully applies deep learning to detect wagons and identify defects with high accuracy. By leveraging **Detectron2** and **Gemini LLM**, we automated defect detection and report generation, making this approach highly effective for real-world applications.

