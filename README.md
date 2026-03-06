# Fraud Detection Using Machine Learning – Full Python Data Science Project (94% Accuracy) 🕵️‍♂️💸

This repository contains a complete, end-to-end Machine Learning project for predicting whether a financial transaction is fraudulent or legitimate. It includes the **Exploratory Data Analysis (EDA)**, **Feature Engineering**, **Model Training**, and a fully functional **web application** built with Streamlit.

## 📊 Exploratory Data Analysis (EDA) & Insights

Before training the model, a thorough analysis of the 6+ million records dataset was conducted in the Jupyter Notebook:

- **Class Imbalance Handling**: Discovered a massive class imbalance (only 0.13% of transactions were frauds). Addressed this during modeling using balanced class weights.
- **Transaction Type Analysis**: Analyzed the distribution of transaction types and identified that frauds **only** occur during `TRANSFER` and `CASH_OUT` operations.
- **Visualizations**: Generated Correlation Matrices (Heatmaps) to find relationships between old/new balances, and used log-scaled histograms to understand the distribution of transaction amounts.
- **Feature Engineering**: Removed non-predictive columns (`step`, `nameOrig`, `nameDest`, `isFlaggedFraud`), applied `StandardScaler` for numerical features, and `OneHotEncoder` for categorical variables.

## 🚀 App Features

- **Real-time Prediction**: A Streamlit web app that instantly classifies new transactions as `Fraud` or `Not Fraud` based on user input.
- **Database Integration**: Automatically logs every transaction checked, the timestamp, and the Logistic Regression model's prediction into a local `transactions.db` SQLite database.
- **Interactive UI**: Clean, user-friendly interface for easy transaction testing.

## 🛠️ Tech Stack

- **Data Science & ML**: Jupyter Notebook, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
- **App Framework**: Streamlit
- **Model Deployment**: Joblib
- **Database**: SQLite3

## 📂 Project Structure

- `notebook.ipynb` - The Jupyter Notebook containing the EDA, data preprocessing, and ML model training.
- `app.py` - The main Streamlit application script for the user interface.
- `fraud_detection_pipeline.pkl` - The pre-trained Logistic Regression Machine Learning model.
- `requirements.txt` - List of Python dependencies required to run the project.
