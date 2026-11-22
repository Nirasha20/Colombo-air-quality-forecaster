# Colombo Air Quality Forecaster 🌫️
**CM4371 Machine Learning & Pattern Recognition – Ensemble Learning Hands-on Assignment**

Predicts **tomorrow's average PM2.5** for **Colombo, Sri Lanka** and compares bagging, boosting, stacking and a hybrid ARIMA + Random Forest model in a Streamlit dashboard.

## Data
- Open-Meteo Air Quality API (CAMS global model): PM2.5, PM10, NO₂, CO
- Open-Meteo Historical Weather API: temperature, humidity, rain, wind
- Hourly data from Aug 2022 to Nov 2026, aggregated to 1,565 daily rows
- Test set: last 12 months, which covers all four Sri Lankan seasons

## Models
| Type | Model |
|---|---|
| Baseline | Persistence (tomorrow = today) |
| Bagging | Random Forest (300 trees) |
| Boosting | Gradient Boosting (400 trees, lr 0.05) |
| Stacking | RF + GB → Ridge meta-model |
| Time series | ARIMA(2,0,1) |
| Hybrid | ARIMA + Random Forest on ARIMA residuals |

## Results (test year)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Hybrid – ARIMA + RF residuals | 3.655 | 4.839 | 0.743 |
| Bagging – Random Forest | 3.724 | 4.890 | 0.737 |
| Stacking – RF + GB → Ridge | 3.745 | 4.938 | 0.732 |
| Boosting – Gradient Boosting | 3.761 | 5.036 | 0.721 |
| ARIMA only | 3.798 | 5.056 | 0.719 |
| Persistence baseline | 3.848 | 5.245 | 0.698 |

The hybrid model was best, about 8% lower RMSE than persistence. See the notebook and dashboard for the explanation.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `01_data.ipynb`: data download, features, models and evaluation
- `app.py`: Streamlit dashboard
- `models.pkl`: trained models
- `results.csv`, `test_predictions.csv`, `colombo_daily.csv`: outputs used by the dashboard

## AI disclosure
Parts of the code in this project were generated with the help of Claude (Anthropic). All code was run, checked and modified by me, and the analysis and interpretation are my own.
