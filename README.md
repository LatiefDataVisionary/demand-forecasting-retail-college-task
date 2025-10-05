# Retail Demand Forecasting Project 📈

![Project Banner](https://via.placeholder.com/1200x300.png?text=Retail+Demand+Forecasting+Model)

A comprehensive machine learning project focused on predicting product demand for a retail business. This repository contains the complete workflow from data ingestion and cleaning to model training and evaluation, developed as a final project for the **Model Development Engineering (Teknik Pengembangan Model)** course.

---

### 📋 Table of Contents
*   [Project Overview](#-project-overview)
*   [Problem Statement](#-problem-statement)
*   [Dataset](#-dataset)
*   [Methodology](#-methodology)
*   [Key Results](#-key-results)
*   [Repository Structure](#-repository-structure)
*   [Setup & Usage](#-setup--usage)
*   [Technologies Used](#-technologies-used)
*   [Contributors](#-contributors)

---

### 🎯 Project Overview
The primary goal of this project is to develop a robust machine learning model that accurately forecasts future product demand. Effective demand forecasting helps businesses optimize inventory, reduce holding costs, prevent stockouts, and improve overall supply chain efficiency. This project explores various time-series features and machine learning models to find the most effective prediction strategy.

### ❓ Problem Statement
A retail company faces challenges in managing its inventory due to unpredictable customer demand. This leads to two critical issues:
1.  **Overstocking:** Excess inventory results in high storage costs and potential waste.
2.  **Understocking:** Insufficient stock leads to lost sales opportunities and decreased customer satisfaction.
The question we aim to answer is: *Can we build a model to reliably predict the units of a product that will be sold on a future date for a specific store?*

### 💾 Dataset
This project integrates three distinct datasets to create a rich feature set for modeling:
*   **Primary Sales Data:** Contains daily sales transactions, including product ID, store ID, and units sold.
*   **Product Information Data:** Contains metadata about each product, such as category, supplier, and type.
*   **External Factors Data:** Includes economic indicators and regional information that may influence sales.

**Note:** The raw datasets are not included in this repository due to their size. To access the data, please refer to [...Link ke Kaggle atau sumber data Anda...]. Place the raw CSV files inside the `data/raw/` directory.

---

### 🛠 Methodology
The project follows a standard machine learning development lifecycle:

1.  **Data Ingestion & Merging:** Loading and merging three different data sources into a single, cohesive dataset.
2.  **Exploratory Data Analysis (EDA):** Analyzing the data to uncover trends, seasonality, and relationships between variables.
3.  **Feature Engineering:** Creating new, meaningful features from the existing data, with a strong focus on time-based features (lags, rolling windows, day-of-week, etc.).
4.  **Data Preprocessing:** Cleaning the data, handling missing values, and encoding categorical variables.
5.  **Model Training:** Training several regression models, including Linear Regression (as a baseline), Random Forest, and XGBoost.
6.  **Model Evaluation:** Evaluating model performance using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on a time-based validation set.
7.  **Conclusion:** Analyzing results and deriving actionable insights.

### 📊 Key Results
After rigorous evaluation, the **XGBoost Regressor** was identified as the best-performing model.

| Model               | RMSE        | MAE         |
|---------------------|-------------|-------------|
| Linear Regression   | [...]       | [...]       |
| Random Forest       | [...]       | [...]       |
| **XGBoost Regressor** | **[...]**   | **[...]**   |

![Prediction vs. Actuals Plot](reports/figures/prediction_vs_actuals.png)
*(Ini adalah contoh, ganti dengan gambar hasil plot Anda sendiri)*

The most important features influencing demand were found to be the product's sales in the previous week (lag features), promotion status, and the time of year (month).

---

### 📂 Repository Structure
```
/
├── data/              # Raw and processed datasets
├── models/            # Trained model files (.pkl)
├── notebooks/         # Jupyter notebooks for analysis and modeling
├── reports/           # Final PDF report, presentation, and figures
├── src/               # Modular Python scripts for data processing and modeling
├── .gitignore         # Specifies files to be ignored by Git
├── LICENSE            # Project license
├── README.md          # This file
└── requirements.txt   # Required Python libraries
```

---

### 🚀 Setup & Usage
To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YourUsername]/demand-forecasting-retail.git
    cd demand-forecasting-retail
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data:**
    *   Download the datasets from [...Link sumber data Anda...] and place them in the `data/raw/` folder.

5.  **Run the notebooks:**
    *   Open the Jupyter notebooks in the `notebooks/` directory and execute them in numerical order.

---

### 💻 Technologies Used
*   **Python 3.9+**
*   **Pandas & NumPy:** for data manipulation and numerical operations.
*   **Scikit-learn:** for data preprocessing and baseline modeling.
*   **XGBoost / LightGBM:** for advanced gradient boosting models.
*   **Matplotlib & Seaborn:** for data visualization.
*   **Jupyter Notebook:** for interactive development and analysis.

---
