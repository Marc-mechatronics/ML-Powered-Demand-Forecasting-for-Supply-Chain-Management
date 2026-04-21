# ML-Powered Demand Forecasting for Supply Chain Management

> A Random Forest regression model that predicts product demand from supply chain variables, deployed as an interactive Flask web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=flat&logo=flask)
![Accuracy](https://img.shields.io/badge/R²%20Score-0.84-green?style=flat)

---

## What This Project Does

In supply chain management, knowing how much of a product to stock is one of the most expensive decisions a company makes. Order too little and you lose sales. Order too much and you waste money on unsold inventory.

This project uses **Machine Learning** to solve that problem. Given a product category, price, current stock level, whether a promotion is running, and the time of year — the model predicts how many units customers will demand.

---

## Demo

Fill in the form → click **Predict Demand** → get an instant forecast.

| Input | Example |
|---|---|
| Product | Electronics |
| Region | North |
| Price | $150 |
| Stock | 80 units |
| Promotion | Active |
| Month | June |
| Day | Monday |
| **Predicted Demand** | **246 units** |

---

## Model Performance

| Metric | Value | Meaning |
|---|---|---|
| R² Score | **0.84** | Model explains 84% of demand variation |
| MAE | **18.6 units** | Average prediction error |
| RMSE | **23.7 units** | Root mean square error |

---

## Tech Stack

- **Python** — core language
- **pandas & NumPy** — data manipulation and generation
- **scikit-learn** — Random Forest model, preprocessing, evaluation
- **matplotlib** — EDA visualisation charts
- **Flask** — web application framework
- **pickle** — model serialisation

---

## Project Structure

```
scm_project/
├── scm_project.py        # Data generation + model training (run first)
├── app.py                # Flask web application (run second)
├── templates/
│   └── index.html        # Web UI
├── static/
│   └── plots/            # EDA charts (auto-generated)
├── data/                 # Dataset CSV (auto-generated)
├── model/                # Saved model files (auto-generated)
└── requirements.txt
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/scm-demand-forecasting.git
cd scm-demand-forecasting
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate data and train the model**
```bash
python scm_project.py
```

**4. Start the web app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

---

## ML Pipeline

```
Data Generation → EDA → Preprocessing → Train/Test Split → Random Forest → Evaluation → Flask Deployment
```

1. **Data Generation** — 1,000 rows with 7 supply chain features and a calculated demand target
2. **EDA** — 6 visualisation charts exploring demand patterns, seasonal trends, and feature correlations
3. **Preprocessing** — LabelEncoder for categorical features, StandardScaler for normalisation
4. **Training** — RandomForestRegressor with 100 trees, 80/20 train-test split
5. **Evaluation** — MAE, RMSE, R² score, Actual vs Predicted scatter plot, Feature Importance
6. **Deployment** — Interactive Flask web app with live prediction form

---

## Why Random Forest?

- Handles both numerical and categorical features naturally
- Captures non-linear relationships between price, promotions, and demand
- Robust to outliers by averaging across 100 decision trees
- Provides feature importance scores showing which variables drive demand most
- Works well out of the box with minimal tuning — ideal for tabular SCM data

---

## Course

Machine Learning Course Project — Supply Chain Management Integration
