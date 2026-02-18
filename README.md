# 📊 Students Performance -- Dashboard de Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-Active-success)

Proyecto End-to-End de Machine Learning en el que analizo datos de
rendimiento académico y construyo modelos predictivos para estimar la
nota en matemáticas utilizando técnicas de Inteligencia Artificial
explicable.

------------------------------------------------------------------------

## 👨‍💻 Sobre Mí

**Andrés Quirós Rojas**\
Estudiante de Ingeniería en Computación\
Instituto Tecnológico de Costa Rica

------------------------------------------------------------------------

## 🎯 Objetivo del Proyecto

El objetivo de este proyecto es simular un pipeline real de Machine
Learning mediante:

-   Análisis Exploratorio de Datos (EDA)\
-   Construcción de pipelines de preprocesamiento\
-   Comparación de modelos de regresión\
-   Aplicación de validación cruzada\
-   Interpretación del comportamiento del modelo con SHAP\
-   Despliegue de un dashboard interactivo con Streamlit\
-   Generación automática de reportes en PDF

------------------------------------------------------------------------

## 📊 Dataset

**Conjunto de datos:** Students Performance Dataset

### 🎯 Variable Objetivo

-   `math score` (nota en matemáticas)

### 📌 Variables Predictoras

-   Nota en lectura\
-   Nota en escritura\
-   Género\
-   Nivel educativo de los padres\
-   Tipo de almuerzo\
-   Curso de preparación para el examen

------------------------------------------------------------------------

## 🧠 Pipeline de Machine Learning

### 🔹 Preprocesamiento de Datos

-   División Train/Test (80/20)\
-   ColumnTransformer\
-   StandardScaler (variables numéricas)\
-   OneHotEncoder (variables categóricas)\
-   Validación Cruzada de 5 folds

### 🔹 Modelos Implementados

-   Regresión Lineal\
-   Random Forest Regressor

### 🔹 Métricas de Evaluación

Cada modelo se evalúa utilizando:

-   R² Score\
-   MAE (Error Absoluto Medio)\
-   RMSE (Raíz del Error Cuadrático Medio)\
-   Promedio de R² en Validación Cruzada

------------------------------------------------------------------------

## 📈 Interpretabilidad del Modelo

Para garantizar transparencia y explicabilidad:

-   Importancia de Variables (Random Forest)\
-   Análisis de valores SHAP\
-   Gráficos de comparación Real vs Predicho

------------------------------------------------------------------------

## 📌 Principales Hallazgos

-   Las notas de lectura y escritura son los predictores más fuertes del
    desempeño en matemáticas.\
-   Los estudiantes que completaron el curso de preparación tienden a
    obtener mejores resultados.\
-   Indicadores socioeconómicos (tipo de almuerzo) influyen en el
    rendimiento académico.\
-   SHAP confirma la relevancia y contribución de las variables
    predictoras principales.

------------------------------------------------------------------------

## 🏗️ Estructura del Proyecto

    students-performance-ml-dashboard/
    │
    ├── app.py
    ├── eda.py
    ├── ml.py
    ├── utils.py
    ├── report.py
    ├── requirements.txt
    ├── StudentsPerformance.csv
    └── README.md

------------------------------------------------------------------------

## 🚀 Cómo Ejecutarlo

### 1️⃣ Clonar el repositorio

``` bash
git clone https://github.com/anquiro20/students-performance-ml-dashboard.git
cd students-performance-ml-dashboard
```

### 2️⃣ Instalar dependencias

``` bash
pip install -r requirements.txt
```

### 3️⃣ Ejecutar la aplicación

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🛠️ Tecnologías Utilizadas

-   Python\
-   Pandas\
-   NumPy\
-   Scikit-learn\
-   SHAP\
-   Plotly\
-   Streamlit\
-   ReportLab

------------------------------------------------------------------------

## 📬 Contacto

Si deseas conectar o conversar sobre este proyecto, no dudes en
contactarme.
