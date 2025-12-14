# Poland Bankruptcy Prediction Dashboard

An end-to-end **machine learning and data visualization project** that predicts the bankruptcy risk of Polish firms using financial indicators.  
The project combines **exploratory data analysis (EDA)**, **model evaluation**, and an **interactive Dash dashboard** for real-time predictions.

---

## Project Overview

This application:

- Explores bankruptcy data for Polish companies
- Trains and evaluates a Gradient Boosting–based classifier
- Visualizes model performance and feature importance
- Allows users to **interactively predict bankruptcy risk** using top financial features

The dashboard is built using **Dash + Plotly**, making it suitable for **deployment and portfolio presentation**.

---

## Model Highlights

- **Algorithm**: Tuned Gradient Boosting Classifier
- **Imbalance handling**: Addressed during training (SMOTE / resampling)
- **Evaluation metrics**:
  - Confusion Matrix
  - ROC Curve (AUC)
  - Precision, Recall, F1-score
- **Explainability**: Feature importance visualization

---

## Dashboard Features

### Exploratory Data Analysis (EDA)
- Target distribution (histogram)
- Class balance (donut chart)
- Correlation matrix for top predictive features

### Model Metrics
- Confusion matrix
- ROC curve with AUC score

### Model Insights
- Top 10 feature importances
- Classification report displayed as a table

### Bankruptcy Prediction Tool
- User inputs top financial indicators
- Outputs:
  - Predicted class (Bankrupt / Not Bankrupt)
  - Probability of bankruptcy

---

## Repository Structure
- Poland-Bankruptcy-Prediction/
  - ├── app.py                         # Dash application
  - ├── final_model.pkl                # Trained ML model
  - ├── dashboard_artifacts.pkl        # Test set & evaluation artifacts
  - ├── poland_bankruptcy_data.json    # Raw dataset
  - ├── requirements.txt               # Python dependencies
  - └── README.md                      # Project documentation

---

## How to Run the App Locally

### Clone the repository
```bash
git clone https://github.com/Bartho-A/Poland-Bankruptcy-Prediction.git
cd Poland-Bankruptcy-Prediction
```
## Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run App
```bash
python app.py
```
2025 Bartholomeow Aobe

