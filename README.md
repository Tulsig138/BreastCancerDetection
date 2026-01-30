Automated Breast Cancer Classification Using ConvNeXt and Transfer Learning
Python TensorFlow Status License

📌 Project Overview
Breast cancer is a leading cause of mortality among women worldwide, and early detection is critical for survival. This project presents an automated deep learning system for classifying mammograms as Benign or Malignant using the CBIS-DDSM dataset.

We leverage ConvNeXt, a modern Convolutional Neural Network, combined with Transfer Learning and a custom ELM-style classification head to achieve high accuracy. Additionally, we integrate Explainable AI (Grad-CAM) to provide visual heatmaps, making the model's decisions interpretable for radiologists.

🚀 Key Features
Advanced Architecture: Utilizes ConvNeXtBase as a feature extractor, outperforming traditional CNNs like ResNet and VGG in global shape recognition.
Efficient Classification: Implements a custom ELM-style Dense Layer with GELU activation for faster convergence and better feature retention.
Explainable AI (XAI): Generates Grad-CAM heatmaps to visualize tumor regions, ensuring trust and transparency in diagnosis.
Robust Preprocessing: Includes metadata mapping, precise resizing, normalization, and data augmentation (rotation, zoom, flips).
High Sensitivity: Optimized to minimize false negatives, achieving a recall rate of 97.66%.
📊 Methodology
The project follows a 4-stage pipeline:

Preprocessing: Data cleaning, metadata mapping of CBIS-DDSM, resizing to 224x224, and normalization.
Augmentation: To prevent overfitting, we apply random rotations, zooms, and horizontal flips.
Feature Extraction: A frozen ConvNeXtBase backbone extracts 1024-dimensional feature vectors.
Classification: Global Average Pooling 
→
 Custom Dense Layer (128 units, GELU) 
→
 Softmax Output.
📈 Results
We benchmarked several state-of-the-art models. ConvNeXtBase achieved the best performance:

Model	Accuracy	Precision	Recall (Sensitivity)	F1-Score
ConvNeXtBase	97.66%	97.66%	97.66%	97.65%
Inception V3	96.56%	96.56%	96.56%	96.56%
DenseNet-121	96.64%	96.66%	95.32%	95.18%
ResNet-50	90.46%	90.21%	89.52%	91.13%
VGG-19	90.42%	89.29%	88.32%	90.16%
🛠️ Tech Stack
Language: Python 3.10
Deep Learning Framework: TensorFlow / Keras
Data Processing: Pandas, NumPy, OpenCV
Visualization: Matplotlib, Seaborn
Environment: Google Colab Pro (NVIDIA T4 GPU)
📂 Dataset
Name: CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
Source: Kaggle / TCIA
Details: The dataset contains verified pathology labels (Benign/Malignant) and ROI masks. Note: You must accept the dataset license on Kaggle/TCIA to use it.
🔧 Installation & Usage
Clone the Repository

git clone [https://github.com/your-username/Breast-Cancer-ConvNeXt.git](https://github.com/your-username/Breast-Cancer-ConvNeXt.git)
cd Breast-Cancer-ConvNeXt
Install Dependencies

pip install tensorflow pandas numpy opencv-python matplotlib seaborn
Run the Notebook

Open Breast_Cancer_Classification.ipynb in Google Colab or Jupyter Notebook.
Ensure the dataset is mounted (update paths in the "Configuration" block).
Run all cells to train the model and generate Grad-CAM visualizations.
👥 Team Members
Krishna Kant Kumar (221FA04735)
Tulsi Gupta (221FA04465)
Guide/Department: Department of Computer Science, Vignan University.

📄 References
L. Wang, "Mammography with Deep Learning for Breast Cancer Detection," Frontiers in Oncology, 2024.
Liu, Z. et al., "A ConvNet for the 2020s," CVPR, 2022.
Selvaraju, R. R. et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV, 2017.
This project was developed as part of the Final Year B.Tech Project at Vignan University.
