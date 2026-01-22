# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/house_price_model.pkl'
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'OverallQual': int(request.form['OverallQual']),
            'GrLivArea': float(request.form['GrLivArea']),
            'TotalBsmtSF': float(request.form['TotalBsmtSF']),
            'GarageCars': float(request.form['GarageCars']),
            'FullBath': int(request.form['FullBath']),
            'Neighborhood': request.form['Neighborhood']
        }

        # Convert to DataFrame (model expects DataFrame)
        df_input = pd.DataFrame([data])

        # Predict
        prediction = model.predict(df_input)[0]

        return render_template('index.html', 
                              prediction_text=f'Estimated House Price: ${prediction:,.0f}',
                              input_data=data)

    except Exception as e:
        return render_template('index.html', 
                              prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)