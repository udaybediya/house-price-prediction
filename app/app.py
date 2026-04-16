import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("house_price_model.pkl")

# Home route (for testing)
@app.route('/')
def home():
    return "Server Running ✅"

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_df = pd.DataFrame([{
            "area_type": data["area_type"],
            "availability": data["availability"],
            "location": data["location"],
            "total_sqft": float(data["sqft"]),
            "bath": int(data["bath"]),
            "bhk": int(data["bhk"])
        }])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "price": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)