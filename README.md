# Hybrid ARFIMA-LSTM Model for Stock Price Prediction

## 📌 Project Overview
This repository contains a **hybrid ARFIMA-LSTM model** designed to predict **S&P 500 Index log returns**. The model integrates **statistical methods (ARFIMA), deep learning (LSTM), and Wavelet Transform** to capture both **long-term dependencies** and **short-term volatility** in financial time series.

The final predictions are obtained using a **Random Forest Regressor**, which combines LSTM outputs for more accurate forecasting.

## 🚀 Methodology
### **1️⃣ Data Collection & Preprocessing**
- **Data Source:** S&P 500 daily closing prices from **Yahoo Finance**.
- **Log Returns Calculation:** \( R_t = \log (P_t / P_{t-1}) \) for variance stabilization.
- **Standardization:** Using mean and standard deviation normalization.

### **2️⃣ ARFIMA Modeling (Long-Term Trends)**
- Captures **long-memory properties** in financial data.
- Extracts **residual errors** for further analysis.

### **3️⃣ Wavelet Transform (Multi-Resolution Analysis)**
- Decomposes residuals into **approximation** and **detail components**.
- **Daubechies db4 wavelet function** is used.

### **4️⃣ LSTM Modeling (Short-Term Dependencies)**
- **One LSTM network** models the **approximation component** (long-term trends).
- **Multiple LSTM networks** model **detail components** (short-term variations).
- Uses **a sliding window approach** for sequential learning.

### **5️⃣ Random Forest Regression (Final Prediction)**
- Combines **LSTM predictions** for the final stock price index forecast.

## 📊 Performance Evaluation
The model is evaluated using the following metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Mean Absolute Percentage Error (MAPE)**
- **Directional Accuracy (%)**

### **🔹 Results**
| Metric | ARFIMA | LSTM | Hybrid (ARFIMA-LSTM) |
|--------|--------|------|----------------------|
| **MSE** | 0.00006 | 0.00006 | **0.00001** |
| **RMSE** | 0.00795 | 0.00805 | **0.00322** |
| **MAE** | 0.00642 | 0.00634 | **0.00268** |
| **R²** | -0.01432 | -0.03923 | **0.83357** |
| **MAPE** | 114.56 | 93.95 | **87.99** |
| **Directional Accuracy** | 71.79% | 56.41% | **89.74%** |

✅ **The hybrid model significantly outperforms ARFIMA and LSTM alone.**

## 📂 Repository Structure
```plaintext
📦 HybridModel
 ┣ 📜 TEL-515_project-code_HybridModel.ipynb  # Jupyter Notebook with the full pipeline
 ┣ 📜 TEL_515_Project_Report.pdf             # Full project report
 ┣ 📜 README.md                               # Project documentation
 ┗ 📜 dataset.csv                             # (If applicable) Sample dataset
```
## 🛠️ Installation & Requirements
To run this project, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn pywt
```

🤝 Contributing
Feel free to fork this repository, submit pull requests, or open issues for improvements.



✉️ Author: Ahmet Kaçmaz
