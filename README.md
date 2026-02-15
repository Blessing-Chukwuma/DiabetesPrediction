# 🏥 Diabetes Risk Prediction System

A machine learning-powered web application built with Streamlit that predicts the risk of diabetes based on patient health metrics. This interactive tool provides real-time predictions, risk assessments, and personalized health recommendations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)
- [Resources](#resources)

---

## 🎯 Overview

The **Diabetes Risk Prediction System** is a comprehensive machine learning application designed to assess the likelihood of a patient having diabetes based on eight key health metrics. Using a trained Neural Network (Multi-Layer Perceptron), the system provides:

- **Real-time predictions** with probability scores
- **Interactive visualizations** of patient health data
- **Risk level classifications** (Low, Moderate, High, Critical)
- **Personalized health recommendations**
- **Model analytics** and performance insights

### 🎓 Project Type
This is a **Capstone Project** demonstrating end-to-end machine learning development from data preprocessing through production deployment.

---

## ✨ Features

### 🔮 Prediction Page
- **Interactive Health Metrics Input**: 8 adjustable sliders for patient data
  - Pregnancies (0-17)
  - Glucose Level (44-199 mg/dL)
  - Blood Pressure (24-122 mmHg)
  - Skin Thickness (7-99 mm)
  - Insulin Level (14-846 mu U/ml)
  - BMI (18.2-67.1 kg/m²)
  - Diabetes Pedigree Function (0.078-2.42)
  - Age (21-81 years)

- **Visual Results Display**:
  - Color-coded prediction boxes (🟢 Low Risk / 🔴 High Risk)
  - Real-time probability score (0-100%)
  - Risk level classification
  - Gauge chart visualization

- **Advanced Visualizations**:
  - Radar chart showing normalized patient values
  - Metrics summary table
  - Risk distribution gauge

- **Health Recommendations**: Context-aware suggestions based on input values

### 📈 Analytics Page
- Model architecture overview
- Training specifications
- Feature distribution histograms
- Model performance metrics
- Model summary details

### ℹ️ About Page
- Detailed system documentation
- Health metrics explanations
- Usage instructions
- Important disclaimers
- Contact and resource links
- Technology stack overview

---

## 📁 Project Structure

```
Diabetes_Capstone Project/
├── README.md                                  # Project documentation (this file)
├── streamlit_app.py                          # Main Streamlit application
├── app (1).py                                 # Alternative implementation
├── requirements.txt                           # Python package dependencies
├── requirements (2).txt                       # Backup requirements file
├── best_mlp (1).keras                        # Trained Neural Network model
├── minmax_scaler (1).pkl                     # Data scaler for preprocessing
├── tuned_logistic_regression_model (1).pkl   # Alternative LR model
└── .venv/                                    # Virtual environment (auto-created)
```

### Key Files Description

| File | Description |
|------|-------------|
| `streamlit_app.py` | Primary application with rich UI and multiple pages |
| `best_mlp (1).keras` | Trained MLP Neural Network model (TensorFlow/Keras) |
| `minmax_scaler (1).pkl` | MinMax scaler for feature normalization |
| `requirements.txt` | All required Python packages and versions |

---

## 🚀 Installation

### Prerequisites
- **Python** 3.8 or higher
- **pip** (Python package manager)
- **Virtual Environment** (recommended)

### Step 1: Clone or Navigate to Project
```bash
cd "c:\Users\chukw\Desktop\Diabetes_Capstone Project"
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- **Data Processing**: numpy, pandas
- **Machine Learning**: scikit-learn, tensorflow, xgboost
- **Web Framework**: streamlit, streamlit-option-menu
- **Visualization**: matplotlib, seaborn, plotly
- **Model Explainability**: shap, lime
- **Utilities**: joblib

### Step 4: Verify Installation
```bash
# Check if streamlit is installed
streamlit --version

# Check if TensorFlow is working
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 📖 Usage

### Running the Application

```bash
streamlit run streamlit_app.py
```

The application will start and display:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Accessing the Application
1. Open your web browser
2. Navigate to **http://localhost:8501**
3. The app will load with the Prediction page active

### Navigation

#### 📊 Prediction Page
1. **Input Patient Data**: Use the sliders to enter health metrics
2. **View Prediction**: Results update in real-time
3. **Analyze Visualizations**: Review radar charts and metrics
4. **Read Recommendations**: Follow health suggestions

#### 📈 Analytics Page
- View model specifications
- Review feature distributions
- Examine model architecture

#### ℹ️ About Page
- Learn about the system
- Understand each health metric
- Access external resources
- Review important disclaimers

### Example Workflow

1. **Scenario**: Assessing a 45-year-old patient
   - Set Pregnancies to 3
   - Set Glucose to 150 mg/dL
   - Set Blood Pressure to 90 mmHg
   - Adjust other metrics as needed
   - View instant prediction and recommendations

---

## 🤖 How It Works

### Prediction Pipeline

```
Patient Input Data
       ↓
MinMax Scaling (Normalization)
       ↓
Neural Network Model
       ↓
Probability Score (0-1)
       ↓
Risk Classification
       ↓
Health Recommendations
```

### Step-by-Step Process

1. **Data Input**: User provides 8 health metrics via interactive sliders
2. **Preprocessing**: Values are normalized using MinMax scaler (0-1 range)
3. **Model Inference**: Preprocessed data passes through trained neural network
4. **Probability Generation**: Model outputs diabetes risk probability
5. **Classification**: 
   - Probability > 0.5 = **High Risk** (Positive Prediction)
   - Probability ≤ 0.5 = **Low Risk** (Negative Prediction)
6. **Risk Levels**:
   - 0-33%: **Low Risk** 🟢
   - 33-66%: **Moderate Risk** 🟡
   - 66-80%: **High Risk** 🔴
   - 80%+: **Critical Risk** 🔴🔴

---

## 🧠 Model Details

### Model Architecture

**Type**: Multi-Layer Perceptron (MLP) Neural Network

**Framework**: TensorFlow/Keras

**Input Features**: 8 health metrics

**Output**: Binary classification (Diabetes / No Diabetes)

**Activation Functions**:
- **Hidden Layers**: ReLU (Rectified Linear Unit)
- **Output Layer**: Sigmoid

**Optimization**: Adam Optimizer

**Loss Function**: Binary Crossentropy

### Model Performance

- **Training**: Optimized for medical diagnosis accuracy
- **Preprocessing**: MinMax scaling (values normalized to 0-1)
- **Threshold**: 0.5 probability

### Model Files

- **best_mlp (1).keras**: Main trained model (Recommended)
- **tuned_logistic_regression_model (1).pkl**: Alternative logistic regression model

---

## 📊 Dataset

### Pima Indians Diabetes Dataset (PIDD)

The model was trained on the Pima Indians Diabetes Dataset, one of the most popular medical ML datasets.

### Feature Specifications

| Feature | Min | Max | Unit | Description |
|---------|-----|-----|------|-------------|
| Pregnancies | 0 | 17 | count | Number of pregnancies |
| Glucose | 44 | 199 | mg/dL | Fasting blood glucose |
| Blood Pressure | 24 | 122 | mmHg | Diastolic blood pressure |
| Skin Thickness | 7 | 99 | mm | Triceps skinfold thickness |
| Insulin | 14 | 846 | mu U/ml | 2-hour serum insulin |
| BMI | 18.2 | 67.1 | kg/m² | Body Mass Index |
| Diabetes Pedigree Function | 0.078 | 2.42 | - | Family history factor |
| Age | 21 | 81 | years | Patient age |

### Target Variable
- **0**: No Diabetes
- **1**: Diabetes Present

---

## 📦 Dependencies

### Core ML Libraries
```
tensorflow==2.13.0       # Deep learning framework
scikit-learn==1.3.0      # ML utilities and preprocessing
xgboost==2.0.0           # Gradient boosting
```

### Data Processing
```
numpy==1.24.3            # Numerical computing
pandas==2.0.3            # Data manipulation
joblib==1.3.1            # Model persistence
```

### Web Framework
```
streamlit==1.28.0        # Web app framework
streamlit-option-menu==0.3.2  # Navigation menu
```

### Visualization
```
matplotlib==3.7.1        # Plotting library
seaborn==0.12.2          # Statistical visualization
plotly==5.17.0           # Interactive plots
```

### Model Explainability
```
shap==0.42.3             # SHAP values
lime==0.2.0              # LIME explanations
```

### Handling Imbalanced Data
```
imbalanced-learn==0.11.0 # Resampling techniques
```

### Install All
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
# Activate virtual environment first
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Issue: "Model file not found" Error

**Solution**:
```bash
# Ensure you're in the correct directory
cd "c:\Users\chukw\Desktop\Diabetes_Capstone Project"

# Verify model files exist
dir best_mlp*.keras
dir minmax_scaler*.pkl
```

### Issue: Port 8501 Already in Use

**Solution 1 - Use different port**:
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Solution 2 - Kill existing process**:
```bash
# Find and kill process using port 8501
lsof -i :8501
kill -9 <PID>
```

### Issue: TensorFlow/CUDA Errors

**Solution**:
```bash
# Reinstall TensorFlow without GPU support
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

### Issue: Slow Predictions or Memory Issues

**Solution**:
```bash
# Reduce model size or increase virtual memory
# Or restart the application
streamlit run streamlit_app.py --logger.level=error
```

### Issue: Plot/Visualization Not Showing

**Solution**:
```bash
# Clear Streamlit cache
streamlit cache clear

# Run app again
streamlit run streamlit_app.py
```

### Issue: Virtual Environment Not Activating

**Windows PowerShell**:
```powershell
# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
.\.venv\Scripts\Activate.ps1
```

---

## ⚠️ Important Disclaimer

### Medical Use Notice

**This tool is for INFORMATIONAL AND EDUCATIONAL PURPOSES ONLY.**

- ❌ **NOT a medical diagnosis tool**: Cannot replace professional medical evaluation
- ❌ **NOT a substitute for doctor consultation**: Always consult healthcare providers
- ❌ **NOT 100% accurate**: ML models have limitations and may error
- ✅ **For awareness only**: Use to understand diabetes risk factors
- ✅ **For educational purposes**: Learn about ML in healthcare

### Limitations

1. **Data-Dependent**: Predictions based on training data characteristics
2. **Boundary Cases**: May perform poorly on extreme values
3. **No Individual Features**: Cannot consider all real-world patient factors
4. **Model Uncertainty**: Confidence varies by input combinations
5. **Static Model**: Doesn't update with new research

### User Responsibility

Users are responsible for:
- Consulting qualified healthcare professionals
- Obtaining proper medical diagnosis
- Following doctor recommendations
- Verifying data accuracy

**In case of health emergency, contact emergency services immediately.**

---

## 📚 Resources

### Diabetes Information
- [American Diabetes Association](https://www.diabetes.org)
- [CDC Diabetes Prevention](https://www.cdc.gov/diabetes)
- [WHO Diabetes Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/diabetes)

### Dataset Source
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)

### Machine Learning Resources
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Docs](https://scikit-learn.org/stable/)

### Related Projects
- SHAP Model Explainability
- LIME Interpretable ML
- XGBoost Gradient Boosting

---

## 📞 Support & Contact

- **Issues**: Report bugs or suggestions
- **Improvements**: Recommendations for features
- **Questions**: Refer to documentation

---

## 📝 License & Attribution

This project uses the Pima Indians Diabetes Dataset.

### Dataset Citation
> National Institute of Diabetes and Digestive and Kidney Diseases. (1999). Pima Indians Diabetes Database. 
> Irvine, CA: University of California, School of Information and Computer Science.

---

## 🎓 Project Information

- **Type**: Capstone Project
- **Focus**: Machine Learning in Healthcare
- **Framework**: Streamlit Web Application
- **Model**: Neural Network (MLP)
- **Created**: 2026
- **Status**: ✅ Fully Functional

---

## 🚀 Quick Start Recap

```bash
# 1. Navigate to project
cd "c:\Users\chukw\Desktop\Diabetes_Capstone Project"

# 2. Activate virtual environment
.\.venv\Scripts\activate

# 3. Install dependencies (if needed)
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py

# 5. Open browser to http://localhost:8501
```

---

## ✅ Verification Checklist

Before using the application, verify:
- [ ] All dependencies installed (`pip list`)
- [ ] Model files present (`best_mlp (1).keras`)
- [ ] Scaler files present (`minmax_scaler (1).pkl`)
- [ ] Streamlit is running without errors
- [ ] Browser shows the application
- [ ] All interactive elements respond

---

**Happy Diabetes Risk Prediction! 🎉**

*Remember: This tool is for awareness and education. Always consult qualified healthcare professionals for medical advice.*

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
