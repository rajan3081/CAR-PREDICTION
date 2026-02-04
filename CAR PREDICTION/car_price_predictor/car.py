# =========================
# IMPORTS
# =========================
import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================
# LOAD DATA
# =========================
car = pd.read_csv(r"C:\Users\lenovo\Downloads\quikr_car.csv")

# =========================
# DATA CLEANING
# =========================
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)

car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)

car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split().str[:3].str.join(' ')
car = car[car['Price'] < 6000000]
car.reset_index(drop=True, inplace=True)

# Save cleaned data
car.to_csv("Cleaned_Car_data.csv", index=False)

print("✅ Data cleaned successfully")

# =========================
# FEATURES & TARGET
# =========================
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# =========================
# TRAIN–TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# ENCODER
# =========================
encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

encoder.fit(X[['name', 'company', 'fuel_type']])

preprocessor = ColumnTransformer(
    [('cat', encoder, ['name', 'company', 'fuel_type'])],
    remainder='passthrough'
)

# =========================
# PIPELINE
# =========================
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)
print("✅ Linear Regression model trained")

# =========================
# R² SCORES
# =========================
# Train R²
train_r2 = model.score(X_train, y_train)

# Test R²
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)

# 🔥 OVERALL R² (FULL DATA)
y_full_pred = model.predict(X)
overall_r2 = r2_score(y, y_full_pred)

print("===================================")
print(f"📈 TRAIN R² SCORE   : {train_r2 * 100:.2f}%")
print(f"🔥 TEST  R² SCORE   : {test_r2 * 100:.2f}%")
print(f"⭐ OVERALL R² SCORE : {overall_r2 * 100:.2f}%")
print("===================================")

# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open("LinearRegressionModel.pkl", "wb"))
print("🎯 Model saved successfully")
