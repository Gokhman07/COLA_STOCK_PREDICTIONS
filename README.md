# Coca-Cola Stock Prediction & Volatility Forecasting

This project analyzes and forecasts **Coca-Cola (KO)** stock behavior using deep learning and econometric time-series models.  
The notebook compares multiple approaches:

- **Several ARIMA specifications** for price forecasting  
- **LSTM Neural Network** for nonlinear prediction  
- **ARCH/GARCH** for volatility forecasting  
- Model evaluation using forecasting error metrics

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
- Daily observations used for training/testing

---

# 1. Price Prediction Models

# ARIMA Model Comparison

Instead of relying on a single ARIMA model, the notebook tests **multiple ARIMA(p,d,q) configurations** to compare forecasting performance.

Examples include different lag and moving-average structures such as:

- ARIMA(1,0,0)
- ARIMA(0,0,1)
- ARIMA(1,1,1)
- Other candidate specifications

This helps identify which structure best matches the dynamics of Coca-Cola prices.

### ARIMA Findings

Different ARIMA models produced different forecast accuracy levels.  
Some specifications performed better than others, showing the importance of model selection rather than choosing ARIMA arbitrarily.

Average forecasting error across tested ARIMA models was approximately:

| Metric | Value |
|---|---:|
| MAPE | ~2.70% |

ARIMA models provided a strong statistical baseline but were less effective than deep learning for nonlinear behavior.

---

# LSTM Neural Network (Best Model)

A Long Short-Term Memory (LSTM) model used the previous **16 closing prices** to predict the next value.

### LSTM Performance

| Metric | Value |
|---|---:|
| RMSE | 1.2368 |
| MAE | 0.9772 |
| MAPE | 1.40% |
| R² | 0.7155 |

The LSTM outperformed the tested ARIMA specifications and produced the strongest price forecasts.

---

# First 10 LSTM Predictions

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

While ARIMA and LSTM focus on prices, ARCH/GARCH models forecast **variance (risk)**.

Financial returns often exhibit:

- Volatility clustering
- Time-varying variance
- Calm and turbulent market periods

---

# GARCH Model Structure

\[
r_t = \mu + \epsilon_t,\quad \epsilon_t \sim N(0,h_t)
\]

\[
h_t = \omega + \alpha \epsilon_{t-1}^2 + \beta h_{t-1}
\]

Where:

- \(r_t\): return  
- \(h_t\): conditional variance  
- \(\alpha\): reaction to shocks  
- \(\beta\): volatility persistence

---

# Interpretation

Typical findings in stock data:

- **High β** → persistent volatility  
- **High α** → strong reaction to new shocks  
- **α + β close to 1** → long memory in risk

This confirms realistic market behavior.

---

# Final Comparison

| Task | Best Model |
|---|---|
| Price Forecasting | LSTM |
| Classical Statistical Forecasting | Best ARIMA Specification |
| Volatility Forecasting | GARCH |

---

# Key Conclusion

This notebook shows that:

- Testing **multiple ARIMA models** improves statistical forecasting quality  
- **LSTM** performs best for nonlinear price prediction  
- **GARCH** is essential for modeling changing market risk  

Using several model families creates a stronger and more realistic forecasting framework.

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

- Automatic ARIMA selection (AIC/BIC)  
- GRU / Transformer models  
- EGARCH / TGARCH  
- Multi-step forecasting  
- Regime-switching models  
- Portfolio risk optimization

---
