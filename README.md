# 📊 Weekly Sales Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-brightgreen)

This project is an end-to-end machine learning system to forecast weekly retail sales using an **XGBoost** model with **92% R² accuracy**. The system is deployed as a user-friendly Streamlit web application that supports real-time single predictions and batch processing via CSV files.

---

## 🛠 Technical Stack

| Category          | Technologies Used                         |
| ---------------   | ----------------------------------------- |
| **Language**      | Python 3.9+                               |
| **ML Model**      | XGBoost                                   |
| **Data Tools**    | Scikit-Learn, Pandas, NumPy               |
| **Optimization**  | RandomizedSearchCV                        |
| **Deployment**    | Streamlit, Joblib                         |
| **Visualization** | Matplotlib, Seaborn                       |

---
---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/weekly-sales-prediction.git](https://github.com/yourusername/weekly-sales-prediction.git)
    cd weekly-sales-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model Files:**
    Download the trained model files (`xgboost_model.pkl`, `scaler.pkl`, `feature_names.json`) and place them in the `models/` directory.

---

## 🔧 How to Use

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    Navigate to `http://localhost:8501` in your web browser.

2.  **Make Predictions:**
    -   **Single Prediction:** Use the sidebar form to enter feature values and click "Predict".
    -   **Batch Prediction:** Upload a CSV file with the required features to get a downloadable file with predictions.

---

## 📊 Model Performance

The XGBoost model was selected after comparing its performance against other models.

| Model             | R-squared Score | MSE (Mean Squared Error) |
| ----------------- | --------------- | ------------------------ |
| **XGBoost**       | **0.92**        | **650,000**              |
| Random Forest     | 0.87            | 800,000                  |
| Linear Regression | 0.75            | 1,200,000                |

---

## 💡 Business Impact

This system enables retailers to:
- ✅ Optimize inventory and reduce waste based on predicted demand.
- ✅ Plan staffing more efficiently around sales forecasts.
- ✅ Design better promotions by understanding key sales drivers.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

