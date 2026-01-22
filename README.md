# House Price Prediction - Ames Housing Dataset

A complete machine learning project that predicts house sale prices in Ames, Iowa using selected features from the famous Kaggle "House Prices: Advanced Regression Techniques" competition.

This project includes:
- Data preprocessing and model training (Linear Regression)
- Model persistence with joblib
- A simple Flask web application with HTML/CSS frontend
- Deployment-ready structure

## Project Requirements Met

- **Selected Features (6 out of allowed 9)**  
  OverallQual, GrLivArea, TotalBsmtSF, GarageCars, FullBath, Neighborhood

- **Algorithm Used**: Linear Regression

- **Web Framework**: Flask + HTML/CSS (no JavaScript frameworks)

- **Model Saving Method**: joblib

- **Evaluation Metrics**: MAE, MSE, RMSE, R²

## Project Structure
HousePrice_Project_EhiomhenTreasure_23CG034059/
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
├── HousePrice_hosted_webGUI_link.txt   # Submission info + live link
│
├── model/
│   ├── model_training.py     # Model building & training script
│   └── house_price_model.pkl # Trained model (created after running training)
│
├── static/
│   └── style.css             # Styling for the web page
│
└── templates/
└── index.html            # Main input form & result page
text## Installation & Local Setup

### Prerequisites

- Python 3.8+
- Kaggle dataset: `train.csv` from [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

### Steps

1. Clone the repository

```bash
git clone [https://github.com/[yourusername]/HousePrice_Project_Adeboyin_[matric].git](https://github.com/treasurebby/HousePrice_Project_TreasureEhiomhen_23CGO34059)
cd HousePrice_Project_EhiomhenTreasure_23CG034059

Install dependencies

Bash pip install -r requirements.txt

Place train.csv in the project root (or update path in model_training.py)
Train & save the model

Bash python model/model_training.py
Expected output includes performance metrics (typical R² ≈ 0.80–0.86 with these features).

Run the web application

Bash python app.py

Open in browser:
http://127.0.0.1:5000/

Features of the Web App

Input fields for the 6 selected features
Clean, responsive form layout
Displays predicted house price in dollars with proper formatting
Basic error handling

Technologies Used

Backend: Flask, scikit-learn, pandas, numpy, joblib
Frontend: HTML5, CSS3
Model: Linear Regression (via scikit-learn Pipeline)
Preprocessing: StandardScaler + OneHotEncoder + ColumnTransformer


License
This project is created for academic purposes (submission on Scorac.com).
Feel free to use it for learning.
