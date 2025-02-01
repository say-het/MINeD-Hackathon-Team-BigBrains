# MINeD-Hackathon-Team-BigBrains

# README: Running the BigBrains Final Project Jupyter Notebook

## Overview
This Jupyter Notebook is designed for running deep learning models using **Detectron2** and other dependencies. It involves image processing, neural network inference, and generating reports with **ReportLab**.

## Prerequisites
Ensure you have the following installed:

- Python 3.8+
- Jupyter Notebook or Jupyter Lab
- Required dependencies (see below)

## Installation
Before running the notebook, install the necessary dependencies by running:

```bash
pip install torch torchvision torchaudio
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html
pip install numpy opencv-python pillow reportlab google-generativeai
```

If using Google Colab, you may need to install Detectron2 manually inside a code cell:

```python
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html
```

## Running the Notebook
1. Open a terminal and navigate to the directory containing the notebook:
   ```bash
   jupyter notebook BigBrains-Final-Project-JK-Lakshmi.ipynb
   ```
2. Run the cells sequentially to process images, run the model, and generate reports.

## Expected Outputs
- The notebook processes images and applies deep learning models.
- Outputs might include visualized images, detection results, and a PDF report using **ReportLab**.

## Troubleshooting
- If you face module import errors, ensure all dependencies are correctly installed.
- Check GPU availability with:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
  If False, consider installing CUDA drivers or running the notebook on a cloud service like Google Colab.

## Notes
- Ensure you have the required dataset and model weights if applicable.
- Modify paths to datasets and output directories as needed in the code cells.

---
This README provides essential information to get started with the notebook. If you encounter issues, consult the official documentation for the respective libraries.

