# MineD | 2025

## 🚀 Manufacturing Use Case Solution With AI

*Developed By:* Team BigBrains, Nirma University  
*For:* JK Lakshmi Cements & Udaipur Cements  

---

## 📌 Problem Statement

### 🚧 Challenges in Rail Wagon Inspection
- *High Dependency on Manual Labor*: Current inspections rely heavily on human effort, making them time-consuming and inconsistent.
- *Risk of Human Error*: Manual verification of wagon numbers and structural damages is prone to inaccuracies.
- *Inefficient Volume Measurement*: Estimating the material volume in loaded wagons lacks precision, leading to inefficiencies.

## 🔍 Our Solution

We have developed an *automated AI-powered inspection system* leveraging *computer vision* to enhance the *safety, accuracy, and efficiency* of rail wagon monitoring. Our system captures real-time images of wagons, analyzes them for:

- *Structural damage detection*
- *Cargo verification*
- *Wagon counting*
- *Damage spot analysis*
- *Automated PDF report generation*

## 🛠 Implementation

### *🔹 Phase 1: Data Preprocessing & Image Extraction*
- Extracted images from raw rail wagon footage (incoming and outgoing wagons).
- Labeled training datasets using *CVAT tool* for supervised learning.

### *🔹 Phase 2: Deep Learning Model Training*
- Leveraged *Detectron2* framework to train a *state-of-the-art image recognition model*.
- Detected:
  - *Damaged parts of wagons*
  - *Cement residue post-loading*
  - *Synthetic images generated by deep learning models*

### *🔹 Phase 3: Wagon Counting Using OpenCV*
- Implemented a *highly accurate wagon counting algorithm* using *OpenCV & NumPy*.
- Achieved a *95% accuracy rate* with a *low loss value of 0.1*.

### *🔹 Phase 4: AI-Powered Report Generation*
- Analyzed images and extracted insights using *Google Gemini AI*.
- Compiled structured, *comprehensive PDF reports* containing:
  - *Damage analysis*
  - *Wagon count & tracking*
  - *Cargo residue identification*

## 🎯 Why Choose Our Solution?

✅ *High Accuracy & Reliability*
- Built on *Detectron2* and *OpenCV* frameworks.
- Delivers *95% accuracy* in wagon counting & defect detection.

✅ *Comprehensive Reporting*
- Generates *detailed PDF reports* with images and AI-generated summaries.
- Enables *quick decision-making* and *compliance verification*.

✅ *Scalability & Automation*
- Fully *automated, reducing dependency on **manual labor*.
- Suitable for large-scale *manufacturing & transport logistics*.

## 🚨 Limitations

🔹 *Day-Night Contrast Issues*: Slight degradation in accuracy due to lighting variations.  
🔹 *Manual Labeling Overhead: Requires **manual dataset preparation* using *CVAT tool*.  
🔹 *Volume Detection Constraints: Further improvement needed in **precise volume estimation*.

## 📂 Demo & Results

🚧 *Project Demonstration*
- 📹 *Video Demonstration: *(To be added)
- 🖼 *Sample Images*: 
 ![WAGON_9](https://github.com/user-attachments/assets/32a53ed1-bd1a-4210-8743-a71c386d3382)

    ![WAGON_1](https://github.com/user-attachments/assets/70d9f035-2dbe-4ecf-9d40-1f55e5feb8af)

    ![Wagon2](https://github.com/user-attachments/assets/39c330aa-fa4f-4463-82e6-08059f051448)

## 📜 *Generated PDF Report*: 
  https://github.com/say-het/MINeD-Hackathon-Team-BigBrains/blob/main/wagonReport_compressed.pdf

## 📌 Tech Stack & References

🔹 *Deep Learning*: Detectron2 (Facebook AI Research)  
🔹 *Computer Vision*: OpenCV, NumPy  
🔹 *AI Text Processing*: Google Gemini AI  
🔹 *Data Labeling*: CVAT  
🔹 *Frameworks*: PyTorch  

## 📑 *References*: 

    

## 🏆 Team BigBrains

🔹 *Het Modi*  
🔹 *Raj Mistry*  
🔹 *Krish Chothani*  
🔹 *Mihir Khunt*  
🔹 *Param Shankar*  

📌 Developed at *Nirma University*  
📌 Part of *JK Lakshmi Cements & Udaipur Cements AI Challenge*

---

🚀 *Contributions & Feedback*  
We welcome feedback and contributions! Feel free to open issues or contribute enhancements to the project.  

💡 *Let's revolutionize industrial AI together!*
