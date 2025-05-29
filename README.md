# 🛒 E-commerce Customer Behavior Big Data Project

This project analyzes customer interaction data on an e-commerce platform using **Apache Spark**, **Kafka**, and **MongoDB**. It performs real-time inference to:

- Identify **funnel drop-offs** in the customer journey  
- Predict **purchase intent** (binary classification)  
- Recommend **top-3 product categories** (multi-label classification)  

---

## 📌 Features

- 🔍 Behavior-Based Purchase Prediction using PySpark and trained ML models  
- 🧠 Multi-label Category Prediction for likely product categories  
- 📉 Funnel Drop-off Analysis with session-based exploration  
- ⚡ Real-Time Streaming Inference using Kafka + Spark + MLflow  
- 🗃 MongoDB Logging with session-level tracking for users  
- 🧪 Manual Grid Search + MLflow for model versioning and selection  
- 🔄 Session Reprocessing Support via log exports and user history  

---

## 🗂️ Project Structure

```
ecommerce-bigdata-project/
│
├── data/                         # Raw dataset (e.g., 2019-Nov.csv)
│
├── notebooks/                    # Exploratory notebooks (EDA, funnel analysis)
│   └── exploration.ipynb
│
├── src/                          # Modular source code
│   ├── preprocessing.py          # Data cleaning & feature engineering
│   ├── model.py                  # ML model training & manual grid search
│   ├── mongo_export.py           # MongoDB session export, fetch, delete
│   ├── streaming_consumer.py     # Kafka consumer + Spark inference
│   └── save_outputs.py           # Save predictions, categories, metadata
│
├── output/                       # Output files
│   ├── predictions.csv           # Binary intent predictions
│   ├── figures/                  # Funnel plots & analytics charts
│   └── category/                 # Multi-label category prediction outputs
│
├── models/                       # MLflow-registered models
│
├── requirements.txt              # Required Python packages
└── README.md                     # Project documentation (this file)
```
