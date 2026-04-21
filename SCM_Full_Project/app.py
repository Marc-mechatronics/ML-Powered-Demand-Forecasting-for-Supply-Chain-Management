"""
================================================
  SCM Demand Forecasting — Web App
  Run AFTER scm_project.py has finished.

  Command: python app.py
  Then open: http://127.0.0.1:5000
================================================
"""

import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load everything the model saved
with open('model/rf_model.pkl',       'rb') as f: model    = pickle.load(f)
with open('model/scaler.pkl',         'rb') as f: scaler   = pickle.load(f)
with open('model/label_encoders.pkl', 'rb') as f: encoders = pickle.load(f)
with open('model/metrics.pkl',        'rb') as f: metrics  = pickle.load(f)

PRODUCTS = ['Electronics','Clothing','Food','Furniture','Toys']
REGIONS  = ['North','South','East','West']

PLOTS = [
    {'file': 'demand_distribution.png', 'title': 'Demand Distribution'},
    {'file': 'demand_by_product.png',   'title': 'Demand by Product'},
    {'file': 'monthly_trend.png',       'title': 'Monthly Trend'},
    {'file': 'promotion_effect.png',    'title': 'Promotion Effect'},
    {'file': 'actual_vs_predicted.png', 'title': 'Actual vs Predicted'},
    {'file': 'feature_importance.png',  'title': 'Feature Importance'},
]

# Home page
@app.route('/')
def index():
    return render_template('index.html',
        products=PRODUCTS, regions=REGIONS,
        metrics=metrics, plots=PLOTS)

# Prediction endpoint — called by the form button
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        product_enc = encoders['product'].transform([data['product']])[0]
        region_enc  = encoders['region'].transform([data['region']])[0]

        features = ['product_enc','region_enc','price','stock',
                    'promotion','month','day_of_week']
        X = pd.DataFrame([[
            product_enc,
            region_enc,
            float(data['price']),
            int(data['stock']),
            int(data['promotion']),
            int(data['month']),
            int(data['day_of_week']),
        ]], columns=features)

        prediction = model.predict(scaler.transform(X))[0]
        return jsonify({'success': True, 'prediction': int(round(prediction))})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("\nWeb app running — open http://127.0.0.1:5000\n")
    app.run(debug=True)
