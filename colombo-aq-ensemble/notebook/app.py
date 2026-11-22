# Colombo Air Quality Forecaster – Ensemble Learning Dashboard
# CM4371 Machine Learning & Pattern Recognition – Hands-on Assignment
#
# AI DISCLOSURE: This dashboard code was generated with the help of Claude (Anthropic)
# and was run, tested and modified by me.
#
# Run locally:  streamlit run app.py
# This file must be in the SAME folder as models.pkl, results.csv,
# test_predictions.csv and colombo_daily.csv (created by the notebook's Save cell).

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
WHO_24H = 15  # WHO 2021 24-hour PM2.5 guideline (µg/m³)

st.set_page_config(page_title="Colombo Air Quality – Ensemble Learning",
                   page_icon="🌫️", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_data():
    d = pd.read_csv(HERE / "colombo_daily.csv", parse_dates=["time"], index_col="time")
    res = pd.read_csv(HERE / "results.csv")
    pr = pd.read_csv(HERE / "test_predictions.csv", parse_dates=["time"], index_col="time")
    return d, res, pr


@st.cache_resource
def load_models():
    return joblib.load(HERE / "models.pkl")


d, res, pr = load_data()
bundle = load_models()
M, F = bundle["models"], bundle["features"]

SEASON_ORDER = ["NE monsoon", "1st inter-monsoon", "SW monsoon", "2nd inter-monsoon"]


def season_of(month):
    if month in (12, 1, 2):
        return "NE monsoon"
    if month in (3, 4):
        return "1st inter-monsoon"
    if month in (5, 6, 7, 8, 9):
        return "SW monsoon"
    return "2nd inter-monsoon"


d["season"] = d.index.month.map(season_of)
pr["season"] = pr.index.month.map(season_of)

res = res.sort_values("RMSE").reset_index(drop=True)
best = res.iloc[0]
base = res[res["Model"].str.startswith("Baseline")].iloc[0]
gain = (base["RMSE"] - best["RMSE"]) / base["RMSE"] * 100
model_cols = [c for c in pr.columns if c not in ("actual", "season")]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🌫️ About")
    st.write("Predicts **tomorrow's average PM2.5** (fine particulate air pollution) "
             "for **Colombo, Sri Lanka** using bagging, boosting, stacking and a "
             "hybrid ARIMA + Random Forest model.")
    st.markdown("**Data:** Open-Meteo air-quality (CAMS) and weather archive APIs, "
                f"{d.index.min():%d %b %Y} – {d.index.max():%d %b %Y} "
                f"({len(d):,} days).")
    st.markdown("**Test set:** last 12 months (all four Sri Lankan seasons).")
    st.divider()
    st.caption("CM4371 – Machine Learning & Pattern Recognition")
    st.caption("AI disclosure: dashboard code generated with help from Claude "
               "(Anthropic), reviewed and modified by the author.")

# ---------------------------------------------------------------- header
st.title("Colombo Air Quality Forecaster")
st.caption("Ensemble learning for next-day PM2.5 – Sri Lankan data")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best model", best["Model"].split("–")[0].strip())
k2.metric("Best test RMSE", f"{best['RMSE']:.2f} µg/m³")
k3.metric("Best R²", f"{best['R2']:.3f}")
k4.metric("Improvement vs persistence", f"{gain:.1f} %")

t1, t2, t3, t4 = st.tabs(["📈 Data & seasons", "🏆 Model comparison",
                          "🔍 Predictions", "🔮 Tomorrow's forecast"])

# ---------------------------------------------------------------- tab 1
with t1:
    fig = px.line(d, y="pm2_5", title="Daily average PM2.5 – Colombo",
                  labels={"pm2_5": "PM2.5 (µg/m³)", "time": ""})
    fig.add_hline(y=WHO_24H, line_dash="dash", annotation_text="WHO 24-h guideline (15)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.box(d, x="month", y="pm2_5", color="season",
                           category_orders={"season": SEASON_ORDER},
                           title="PM2.5 by month",
                           labels={"pm2_5": "PM2.5 (µg/m³)"}),
                    use_container_width=True)
    c2.plotly_chart(px.scatter(d, x="precipitation", y="target", color="season",
                               opacity=0.5, category_orders={"season": SEASON_ORDER},
                               title="Today's rain vs tomorrow's PM2.5",
                               labels={"precipitation": "Rain today (mm)",
                                       "target": "PM2.5 tomorrow (µg/m³)"}),
                    use_container_width=True)

    st.info("**What the data shows:** PM2.5 is highest from December to April. During the "
            "Northeast monsoon, northerly winds carry pollution from the Indian subcontinent, "
            "and in the first inter-monsoon (Mar–Apr) weak winds let it build up. It is lowest "
            "in the Southwest monsoon (May–Sep), when rain washes particles out and clean "
            "winds blow in from the Indian Ocean.")

# ---------------------------------------------------------------- tab 2
with t2:
    st.subheader("Test-set results (last 12 months)")
    st.dataframe(res.round(3), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    fig = px.bar(res.sort_values("RMSE", ascending=False), x="RMSE", y="Model",
                 orientation="h", title="Test RMSE (lower is better)", text_auto=".2f")
    fig.update_xaxes(range=[res["RMSE"].min() * 0.9, res["RMSE"].max() * 1.03])
    c1.plotly_chart(fig, use_container_width=True)

    tt = res.melt(id_vars="Model", value_vars=["Train_RMSE", "RMSE"],
                  var_name="Set", value_name="RMSE (µg/m³)")
    tt["Set"] = tt["Set"].map({"Train_RMSE": "Train", "RMSE": "Test"})
    c2.plotly_chart(px.bar(tt, x="Model", y="RMSE (µg/m³)", color="Set", barmode="group",
                           title="Train vs test error (overfitting check)"),
                    use_container_width=True)

    # ✏️ Rewrite this explanation in your own words – it is the most important part of the marks.
    st.subheader("Why the models behaved this way")
    st.markdown(f"""
- **Hybrid ARIMA + RF was best** (RMSE {best['RMSE']:.2f}). ARIMA captures the linear
  day-to-day carry-over of pollution, and the Random Forest learns ARIMA's errors from
  weather (rain, wind direction, season). The two parts make *different* kinds of errors,
  so combining them helps.
- **Bagging (Random Forest) beat boosting (Gradient Boosting).** The dataset is small and
  noisy (about 1,200 training days). Boosting fits trees one after another to the remaining
  error, so it also fits noise. Its training error is very low but its test error is higher.
  Averaging 300 independent trees reduces variance, so RF generalises better.
- **Stacking did not beat RF on its own.** Its base models (RF and GB) are both tree ensembles
  on the same features, so they make very similar errors and the Ridge meta-model has little
  new information to combine.
- **All tree models overfit to some degree** (train RMSE ≈ 2.4 vs test ≈ 4.9). ARIMA has
  equal train and test error: it is stable but too simple to use weather information.
- **Every model beats the persistence baseline**, but only by about {gain:.0f}%. Colombo's
  PM2.5 changes slowly from day to day, so "tomorrow ≈ today" is already a strong forecast.
- The top three models are within about 0.1 µg/m³ of each other, so the differences
  between them are small.
""")

# ---------------------------------------------------------------- tab 3
with t3:
    default = [m for m in model_cols if m.startswith(("Hybrid", "Baseline"))]
    pick = st.multiselect("Models to show", model_cols, default=default)
    fig = px.line(pr, y=["actual"] + pick, title="Actual vs predicted PM2.5 – test year",
                  labels={"value": "PM2.5 (µg/m³)", "time": "", "variable": ""})
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    err = pd.DataFrame({m: (pr["actual"] - pr[m]).abs() for m in model_cols})
    err["season"] = pr["season"]
    es = err.groupby("season").mean().reindex(SEASON_ORDER).reset_index()
    es = es.melt(id_vars="season", var_name="Model", value_name="MAE (µg/m³)")
    c1.plotly_chart(px.bar(es, x="season", y="MAE (µg/m³)", color="Model", barmode="group",
                           title="Average error by season"),
                    use_container_width=True)

    hide_today = c2.checkbox("Hide today's PM2.5 (shows other features more clearly)")
    imp = pd.Series(M["Bagging – Random Forest"].feature_importances_, F).sort_values()
    if hide_today:
        imp = imp.drop("pm2_5")
    c2.plotly_chart(px.bar(imp.tail(12), orientation="h",
                           title="Random Forest – feature importance",
                           labels={"value": "Importance", "index": ""}).update_layout(
                               showlegend=False),
                    use_container_width=True)

# ---------------------------------------------------------------- tab 4
with t4:
    last_day = d.index[-1]
    x = d[F].iloc[[-1]].copy()
    st.subheader(f"Forecast for {last_day + pd.Timedelta(days=1):%A %d %B %Y}")
    st.caption(f"Based on observed conditions on {last_day:%d %B %Y}. "
               "Use the sliders to test 'what-if' scenarios.")

    s1, s2 = st.columns(2)
    x["pm2_5"] = s1.slider("Today's PM2.5 (µg/m³)", 0.0, 80.0, float(x["pm2_5"].iloc[0]), 0.5)
    x["precipitation"] = s2.slider("Today's rain (mm)", 0.0, 100.0,
                                   float(x["precipitation"].iloc[0]), 0.5)

    out = {
        "Bagging – Random Forest": M["Bagging – Random Forest"].predict(x[F])[0],
        "Boosting – Gradient Boosting": M["Boosting – Gradient Boosting"].predict(x[F])[0],
        "Stacking – RF + GB → Ridge": M["Stacking – RF + GB → Ridge"].predict(x[F])[0],
        "Hybrid – ARIMA + RF residuals": d["arima_next"].iloc[-1] + M["rf_residual"].predict(x[F])[0],
    }
    cols = st.columns(4)
    for col, (name, val) in zip(cols, out.items()):
        col.metric(name, f"{val:.1f} µg/m³")

    v = out["Hybrid – ARIMA + RF residuals"]
    if v <= WHO_24H:
        st.success(f"Best model forecast {v:.1f} µg/m³ – within the WHO 24-hour guideline (15).")
    elif v <= 37.5:
        st.warning(f"Best model forecast {v:.1f} µg/m³ – above the WHO guideline; "
                   "sensitive groups (asthma, elderly, children) should take care.")
    else:
        st.error(f"Best model forecast {v:.1f} µg/m³ – high; limit outdoor activity if sensitive.")
    st.caption("Note: the ARIMA part of the hybrid uses the observed series, so it does not "
               "react to the sliders; only the RF correction does.")
