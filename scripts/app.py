from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("sales_forecast_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Rossmann Sales Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data
        data = request.get_json()  # Expects JSON input
        input_df = pd.DataFrame([data])  # Convert to DataFrame
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Return the result
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)  # Run in debug mode for local testing
