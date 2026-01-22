# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# ── CONFIG ────────────────────────────────────────────────
SELECTED_FEATURES = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'Neighborhood']
TARGET = 'SalePrice'
MODEL_PATH = 'house_price_model.pkl'

# ── 1. Load data ──────────────────────────────────────────
# You need to download train.csv from Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
df = pd.read_csv('train.csv')   # ← change path if needed

print("Dataset shape:", df.shape)

# ── 2. Preprocessing ──────────────────────────────────────
X = df[SELECTED_FEATURES].copy()
y = df[TARGET].copy()

# Handle missing values (very few in these columns)
print("Missing values:\n", X.isnull().sum())
X['TotalBsmtSF'] = X['TotalBsmtSF'].fillna(0)     # most common
X['GarageCars'] = X['GarageCars'].fillna(0)
# others should have almost no missing

# ── 3. Train-test split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Preprocessing pipeline ─────────────────────────────
numeric_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath']
categorical_features = ['Neighborhood']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# ── 5. Full pipeline ──────────────────────────────────────
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# ── 6. Train ──────────────────────────────────────────────
model.fit(X_train, y_train)

# ── 7. Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance on Test Set:")
print(f"MAE:  {mae:,.0f}")
print(f"MSE:  {mse:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²:   {r2:.4f}")

# Typical performance with these features: R² ≈ 0.80–0.86

# ── 8. Save model ─────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(model, os.path.join('model', MODEL_PATH))
print(f"Model saved to: model/{MODEL_PATH}")