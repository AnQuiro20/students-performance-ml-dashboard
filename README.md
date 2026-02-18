# 📊 Students Performance – Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Active-success)

End-to-End Machine Learning project where I analyze academic performance data and build predictive models to estimate math scores using explainable AI techniques.

---

## 👨‍💻 About Me

**Andrés Quirós Rojas**  
Computer Engineering Student  
Instituto Tecnológico de Costa Rica  

---

## 🎯 Project Objective

The objective of this project is to simulate a real-world Machine Learning pipeline by:

- Performing Exploratory Data Analysis (EDA)
- Building preprocessing pipelines
- Comparing regression models
- Applying cross-validation
- Interpreting model behavior using SHAP
- Deploying an interactive dashboard with Streamlit
- Generating automated PDF performance reports

---

## 📊 Dataset

**Dataset:** Students Performance Dataset  

### 🎯 Target Variable
- `math score`

### 📌 Predictor Variables
- Reading score  
- Writing score  
- Gender  
- Parental level of education  
- Lunch type  
- Test preparation course  

---

## 🧠 Machine Learning Pipeline

### 🔹 Data Preprocessing

- Train/Test Split (80/20)
- ColumnTransformer
- StandardScaler (numerical features)
- OneHotEncoder (categorical features)
- 5-Fold Cross Validation

### 🔹 Models Implemented

- Linear Regression  
- Random Forest Regressor  

### 🔹 Evaluation Metrics

Each model is evaluated using:

- R² Score  
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- Cross-Validation Mean R²  

---

## 📈 Model Interpretability

To ensure transparency and explainability:

- Feature Importance (Random Forest)
- SHAP values analysis
- Actual vs Predicted comparison plots

This allows better understanding of how different variables impact student performance predictions.

---

## 📌 Key Insights

- Reading and writing scores are the strongest predictors of math performance.
- Students who completed test preparation courses tend to perform better.
- Socioeconomic indicators (lunch type) influence academic outcomes.
- SHAP confirms the relevance and contribution of main predictive features.

---

## 🏗️ Project Structure

```
students-performance-ml-dashboard/
│
├── app.py              # Streamlit dashboard
├── eda.py              # Exploratory analysis functions
├── ml.py               # ML pipeline and model comparison
├── utils.py            # Logging and dataset loading
├── report.py           # Automated PDF report generation
├── requirements.txt
├── StudentsPerformance.csv
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/anquiro20/students-performance-ml-dashboard.git
cd students-performance-ml-dashboard
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application

```bash
streamlit run app.py
```

The dashboard will open locally in your browser.

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- SHAP  
- Plotly  
- Streamlit  
- ReportLab  

---

## 💡 What This Project Demonstrates

Through this project, I demonstrate my ability to:

- Design modular Machine Learning architectures  
- Apply structured preprocessing pipelines  
- Compare and evaluate regression models properly  
- Use explainable AI techniques  
- Build interactive dashboards  
- Translate technical results into actionable insights  

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Model persistence using joblib  
- Cloud deployment (Streamlit Cloud / Render)  
- Docker containerization  
- Testing with larger and more complex datasets  

---

## 📬 Contact

If you would like to connect or discuss this project, feel free to reach out.
