"""
================================================
  SCM Demand Forecasting — Main Project File
  Run this file ONCE to generate data + train
  the model. Then run app.py for the website.

  Command: python scm_project.py
================================================
"""

import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import mean_absolute_error, mean_squared_error


# ════════════════════════════════════════════════════════════════════
#  PART 1 — GENERATE DATA
#  Creates a realistic supply chain dataset (1000 rows)
#  and saves it to data/scm_demand.csv
# ════════════════════════════════════════════════════════════════════

def generate_data():
    print("\n[ STEP 1 ] Generating dataset...")

    os.makedirs('data', exist_ok=True)

    np.random.seed(42)
    n = 1000

    products  = np.random.choice(['Electronics','Clothing','Food','Furniture','Toys'], n)
    regions   = np.random.choice(['North','South','East','West'], n)
    prices    = np.round(np.random.uniform(10, 500, n), 2)
    stock     = np.random.randint(0, 200, n)
    promotion = np.random.choice([0, 1], n)   # 0 = no promo, 1 = promo active
    month     = np.random.randint(1, 13, n)   # 1 to 12
    day       = np.random.randint(1, 8, n)    # 1=Monday to 7=Sunday

    # Demand formula: price reduces it, promotion and stock increase it
    demand = 200 - 0.3 * prices + 50 * promotion + 0.5 * stock
    demand = demand + 20 * np.sin(2 * 3.14 * month / 12)  # seasonal wave
    demand = demand + np.random.normal(0, 20, n)            # random noise
    demand = np.clip(np.round(demand), 0, 600).astype(int)

    df = pd.DataFrame({
        'product':     products,
        'region':      regions,
        'price':       prices,
        'stock':       stock,
        'promotion':   promotion,
        'month':       month,
        'day_of_week': day,
        'demand':      demand,
    })

    df.to_csv('data/scm_demand.csv', index=False)
    print(f"    Done — {len(df)} rows saved to data/scm_demand.csv")
    return df


# ════════════════════════════════════════════════════════════════════
#  PART 2 — TRAIN THE MODEL
#  Loads the CSV, draws EDA charts, preprocesses, trains a Random
#  Forest, evaluates it, and saves everything to model/
# ════════════════════════════════════════════════════════════════════

def train_model(df):
    print("\n[ STEP 2 ] Training the model...")

    os.makedirs('static/plots', exist_ok=True)
    os.makedirs('model',        exist_ok=True)

    # ── EDA: draw charts and save them ───────────────────────────────

    # Chart 1 — how demand is spread
    plt.figure(figsize=(7, 3))
    plt.hist(df['demand'], bins=30, color='#2563EB', edgecolor='white')
    plt.title('Demand Distribution')
    plt.xlabel('Demand (units)')
    plt.tight_layout()
    plt.savefig('static/plots/demand_distribution.png', dpi=100)
    plt.close()

    # Chart 2 — average demand per product
    plt.figure(figsize=(7, 3))
    df.groupby('product')['demand'].mean().sort_values().plot(kind='barh', color='#2563EB')
    plt.title('Average Demand by Product')
    plt.xlabel('Average Demand')
    plt.tight_layout()
    plt.savefig('static/plots/demand_by_product.png', dpi=100)
    plt.close()

    # Chart 3 — demand across months
    plt.figure(figsize=(7, 3))
    df.groupby('month')['demand'].mean().plot(color='#2563EB', marker='o')
    plt.title('Average Demand by Month')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.savefig('static/plots/monthly_trend.png', dpi=100)
    plt.close()

    # Chart 4 — promotion effect
    plt.figure(figsize=(5, 3))
    df.groupby('promotion')['demand'].mean().plot(
        kind='bar', color=['#94A3B8', '#2563EB'], rot=0)
    plt.xticks([0, 1], ['No Promo', 'With Promo'])
    plt.title('Promotion Impact on Demand')
    plt.tight_layout()
    plt.savefig('static/plots/promotion_effect.png', dpi=100)
    plt.close()

    print("    EDA charts saved.")

    # ── Preprocessing ─────────────────────────────────────────────────

    # Convert text columns to numbers (ML needs numbers)
    le_product = LabelEncoder()
    le_region  = LabelEncoder()
    df['product_enc'] = le_product.fit_transform(df['product'])
    df['region_enc']  = le_region.fit_transform(df['region'])

    # Save encoders so app.py can use them
    with open('model/label_encoders.pkl', 'wb') as f:
        pickle.dump({'product': le_product, 'region': le_region}, f)

    # Input features and target
    features = ['product_enc','region_enc','price','stock',
                'promotion','month','day_of_week']
    X = df[features]
    y = df['demand']

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale to the same range
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # ── Train ─────────────────────────────────────────────────────────

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_sc, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────

    y_pred = model.predict(X_test_sc)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot

    print(f"    MAE : {mae:.2f} units")
    print(f"    RMSE: {rmse:.2f} units")
    print(f"    R2  : {r2:.4f}  (1.0 = perfect)")

    # Chart 5 — actual vs predicted
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.4, color='#2563EB', s=20)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect fit')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title(f'Actual vs Predicted  (R2={r2:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/plots/actual_vs_predicted.png', dpi=100)
    plt.close()

    # Chart 6 — feature importance
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(7, 3))
    imp.plot(kind='barh', color='#2563EB')
    plt.title('Which features matter most?')
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=100)
    plt.close()

    # ── Save everything ───────────────────────────────────────────────

    with open('model/rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    metrics = {'mae': round(mae,2), 'rmse': round(rmse,2), 'r2': round(r2,4)}
    with open('model/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print("    Model saved to model/rf_model.pkl")
    print("\n[ DONE ] Now run: python app.py")
    return metrics


# ════════════════════════════════════════════════════════════════════
#  MAIN — runs both parts when you do: python scm_project.py
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 50)
    print("  SCM Demand Forecasting — ML Project")
    print("=" * 50)

    df = generate_data()
    train_model(df)
