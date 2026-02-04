import pandas as pd
import pickle
from flask import Flask, render_template, request

# =========================
# LOAD MODEL & DATA
# =========================
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
car = pd.read_csv("Cleaned_Car_data.csv")

app = Flask(__name__)

# =========================
# HOME
# =========================
@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    company_models = {
    c: sorted(car[car['company'] == c]['name'].unique())
    for c in companies
}


    return render_template(
        "index.html",
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        company_models=company_models
    )

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_df = pd.DataFrame([{
            'name': request.form['car_models'],
            'company': request.form['company'],
            'year': int(request.form['year']),
            'kms_driven': int(request.form['kilo_driven']),
            'fuel_type': request.form['fuel_type']
        }])

        prediction = model.predict(input_df)[0]
        return str(int(round(prediction)))

    except Exception as e:
        return f"Error: {e}", 500

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
