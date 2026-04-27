# Coca-Cola Stock Prediction & Volatility Forecasting

This project analyzes and forecasts **Coca-Cola (KO)** stock behavior using both deep learning and econometric time-series models.  
The notebook combines:

- **LSTM Neural Network** for stock price prediction  
- **ARIMA** for baseline forecasting  
- **ARCH/GARCH** for volatility forecasting  
- Performance evaluation using multiple metrics

---

# Project Goals

The objective is to compare models for two important financial tasks:

1. **Predict future stock prices**
2. **Estimate future market volatility (risk)**

This reflects real-world finance, where returns and volatility are often modeled separately.

---

# Dataset

- Asset: **Coca-Cola (KO)**
- Variable: Daily Closing Prices
- Source: Yahoo Finance
- Sample Size: 100+ observations

---

# 1. Price Prediction Models

## ARIMA Baseline Model

A classical ARIMA model was fitted as a benchmark time-series model.

### ARIMA Performance

| Metric | Value |
|---|---:|
| MAPE | ~2.70% |

ARIMA provided reasonable short-term forecasts but struggled to capture nonlinear price dynamics.

---

## LSTM Neural Network (Best Model)

A Long Short-Term Memory (LSTM) model used the previous **16 closing prices** to predict the next value.

### LSTM Performance

| Metric | Value |
|---|---:|
| RMSE | 1.2368 |
| MAE | 0.9772 |
| MAPE | 1.40% |
| R² | 0.7155 |

The LSTM outperformed ARIMA and produced the most accurate forecasts.

---

## First 10 LSTM Predictions

| Date | Actual | Predicted | Error |
|---|---:|---:|---:|
| 2025-09-09 | 66.8738 | 67.5138 | -0.6400 |
| 2025-09-10 | 66.8344 | 67.3223 | -0.4879 |
| 2025-09-11 | 66.6373 | 67.1549 | -0.5176 |
| 2025-09-12 | 66.0362 | 67.0143 | -0.9781 |
| 2025-09-15 | 65.7482 | 66.8617 | -1.1135 |
| 2025-09-16 | 65.7780 | 66.6779 | -0.8999 |
| 2025-09-17 | 66.5724 | 66.4823 | 0.0901 |
| 2025-09-18 | 65.9965 | 66.3405 | -0.3440 |
| 2025-09-19 | 65.9667 | 66.2412 | -0.2745 |
| 2025-09-22 | 65.7482 | 66.1579 | -0.4097 |

---

# 2. Volatility Forecasting with ARCH/GARCH

While LSTM predicts prices, ARCH/GARCH models forecast **variance (risk)**.

This is important because financial returns often exhibit:

- Volatility clustering
- Time-varying variance
- Calm and turbulent market regimes

---

## GARCH Model Structure

\[
r_t = \mu + \epsilon_t,\quad \epsilon_t \sim N(0,h_t)
\]

\[
h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}
\]

Where:

- \(r_t\): return  
- \(h_t\): conditional variance  
- \(\alpha\): shock impact  
- \(\beta\): persistence of volatility

---

## Interpretation of Results

If the notebook output shows:

- **High β** → volatility is persistent  
- **High α** → market reacts strongly to shocks  
- **α + β close to 1** → long memory in volatility

This behavior is common in real financial markets.

---

# Final Model Comparison

| Task | Best Model |
|---|---|
| Price Forecasting | LSTM |
| Classical Baseline | ARIMA |
| Volatility Forecasting | GARCH |

---

# Key Conclusion

This notebook demonstrates that:

- **LSTM** is stronger for nonlinear price prediction  
- **ARIMA** is useful as an interpretable benchmark  
- **GARCH** is essential for modeling risk and volatility  

Using multiple model families provides a more complete financial forecasting framework.

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras  
- Statsmodels  
- ARCH package  
- Yahoo Finance API

---

# Future Improvements

- GRU / Transformer models  
- EGARCH / TGARCH for asymmetric shocks  
- Multi-step forecasting  
- Regime-switching models  
- Portfolio risk optimization

---
