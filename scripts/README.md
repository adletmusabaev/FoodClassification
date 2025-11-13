🍽️ Food Classification - SOTA Computer Vision Demo
Assignment 2 - Computer Vision
Astana IT University

📋 Project Overview
This project implements a state-of-the-art food classification system using EfficientNetV2 architecture with Grad-CAM explainability. The system can classify food images into 4 categories: Bread, Fried Food, Seafood, and Vegetable-Fruit.
✨ Key Features

🧠 SOTA Model: EfficientNetV2 architecture
🔥 Explainability: Grad-CAM heatmaps showing model focus areas
📸 Multiple Input Methods: Upload images or use camera
⚡ Real-time Classification: Fast inference with latency tracking
📊 Comprehensive Metrics: Top-1, Top-2 accuracy, confusion matrices
🎨 Modern Web Interface: Responsive design with beautiful UI


📁 Project Structure
FoodClassification/
├── dataset/
│   ├── train/
│   │   ├── Bread/
│   │   ├── Fried food/
│   │   ├── Seafood/
│   │   └── Vegetable-Fruit/
│   └── test/
│       └── (same structure)
├── models/
│   ├── baseline_cnn.h5
│   └── transfer_mobilenetv2.h5
├── models_sota/
│   ├── best_model.h5
│   └── final_sota_model.h5
├── results_sota/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── comparison_results/
│   ├── complete_comparison.png
│   └── comparison_table.csv
├── split_dataset.py
├── train_baseline.py
├── train_transfer.py
├── dataset_analysis.py
├── synthetic_data_prep.py
├── art_augmentation.py
├── sota_training.py
├── ablation_study.py
├── model_comparison.py
├── flask_backend.py
├── index.html
├── run_demo.py
└── README.md

🚀 Installation
1. Clone or Download Project
bashcd FoodClassification
2. Install Dependencies
bashpip install tensorflow
pip install flask flask-cors
pip install opencv-python
pip install pillow
pip install numpy pandas
pip install matplotlib seaborn
pip install scikit-learn
pip install albumentations
Or use requirements.txt:
bashpip install -r requirements.txt

📊 Dataset Preparation
Step 1: Collect Data
Place your food images in respective category folders:
dataset/
├── Bread/
├── Fried food/
├── Seafood/
└── Vegetable-Fruit/
Step 2: Split Dataset
bashpython split_dataset.py
This creates train/test split (80/20).
Step 3: Analyze Dataset
bashpython dataset_analysis.py
Step 4: Generate Synthetic Data (Optional)
bashpython synthetic_data_prep.py
This generates prompts for data augmentation using Stable Diffusion or similar tools.
Step 5: Apply Advanced Augmentation
bashpython art_augmentation.py

🎓 Model Training
Baseline Model (Assignment 1)
bash# Simple CNN
python train_baseline.py

# Transfer Learning (MobileNetV2)
python train_transfer.py
SOTA Model (Assignment 2)
bash# Train EfficientNetV2 with advanced augmentation
python sota_training.py
This will:

Train for ~30 epochs with two-phase training
Generate training curves
Create confusion matrix
Save best model

Ablation Study
bashpython ablation_study.py
Compare with/without augmentation.
Model Comparison
bashpython model_comparison.py
Compare all three models (Baseline, Transfer, SOTA).

🌐 Web Application
Quick Start
bashpython run_demo.py
This will:

Check dependencies
Load trained model
Start Flask backend on port 5000
Open browser automatically

Manual Start
Terminal 1: Backend
bashpython flask_backend.py
Terminal 2: Frontend
Open index.html in browser or navigate to http://localhost:5000

📖 Usage Guide
Web Interface

Upload Image: Click "Choose Image" and select a food photo
Use Camera: Click "Use Camera" for real-time capture
View Results: See prediction with confidence scores
Check Heatmap: Grad-CAM shows where model focused

API Usage
Health Check
bashcurl http://localhost:5000/health
Predict
pythonimport requests
import base64

# Read image
with open('food.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post('http://localhost:5000/predict',
    json={'image': f'data:image/jpeg;base64,{img_data}'})

result = response.json()
print(result['top_prediction'])

📊 Results
Model Performance
ModelAccuracyInference TimeParametersBaseline CNN~75%~15ms1.2MMobileNetV2~85%~20ms2.3MEfficientNetV2 (SOTA)~92%~35ms5.9M
Note: Actual results depend on your dataset
Key Improvements

+17% accuracy over baseline
Grad-CAM explainability for trust
Advanced augmentation for robustness
Two-phase training for better generalization


🎯 SOTA Methods Used
1. EfficientNetV2

State-of-the-art CNN architecture
Better accuracy with fewer parameters
Faster training than EfficientNetV1

2. Advanced Data Augmentation (ArtAug concept)

Rotation, zoom, shift
Brightness/contrast adjustment
Category-specific augmentations
Composition improvements

3. Synthetic Data Generation (TA-TiTok concept)

Text-to-image prompts for rare classes
Diverse lighting/plating variations
Balance dataset distribution

4. Grad-CAM Explainability

Visualize model attention
Build trust in predictions
Debug misclassifications


 Report Checklist
For your Assignment 2 report, include:

 Dataset statistics (size, balance, synthetic %)
 SOTA method explanation (1 page per method)
 Architecture diagram
 Training curves
 Metrics tables (Baseline vs SOTA)
 Ablation study results
 Grad-CAM visualizations
 Confusion matrices
 Ethics & limitations discussion
 Reproducibility instructions


 Demo Video
Record 3-5 minute video showing:

Dataset overview (show examples from each class)
Training process (show training curves)
Web app demo (upload image, show predictions)
Grad-CAM explanation (explain heatmap)
Model comparison (show metrics table)
Ablation study (impact of augmentation)


 Troubleshooting
Model Not Loading
bash# Check model file exists
ls models_sota/final_sota_model.h5

# Try with absolute path
python flask_backend.py
Port Already in Use
bash# Kill process on port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:5000 | xargs kill -9
Camera Not Working

Allow camera permissions in browser
Use HTTPS or localhost
Check browser console for errors

Low Accuracy

Collect more training data
Apply more augmentation
Train for more epochs
Try different hyperparameters


🎓 Assignment Requirements Met
✅ Dataset: Collected and split
✅ Baseline: Simple CNN trained (A1)
✅ Transfer Learning: MobileNetV2 trained (A1)
✅ SOTA Method: EfficientNetV2 with augmentation
✅ Web App: Upload + Camera + Predictions
✅ Explainability: Grad-CAM heatmaps
✅ Metrics: Accuracy, F1, confusion matrix
✅ Ablation: With/without augmentation
✅ Comparison: All models compared
✅ Ethics: Synthetic data labeled

📚 References

EfficientNetV2: https://arxiv.org/abs/2104.00298
Grad-CAM: https://arxiv.org/abs/1610.02391
ArtAug concept: Neurohive
TA-TiTok concept: Neurohive





🎉 Credits
Assignment 2 - Computer Vision
Instructor: Baimukanova Zhanerke
Astana IT University



⚖️ License
This project is for educational purposes (Assignment 2).
All synthetic images are clearly labeled and watermarked.

Due Date: November 13, 2025 at 23:59 (Asia/Almaty)
Good luck! 