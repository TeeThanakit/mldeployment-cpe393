from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # check feature is a key
    if "features" not in data:
        return jsonify({"error": '"features" key is missing'}), 400

    features = data["features"]

    # check input 4 float values
    if not isinstance(features, list):
        return jsonify({"error": '"features" must be a list'}), 400
    
    # print("Log:", features)
    
    for i, item in enumerate(features):
        if not (isinstance(item, list) and len(item) == 4):
            return jsonify({"error": f"Each feature set must be a list of exactly 4 numbers (problem at index {i})"}), 400
        if not all(isinstance(val, (int, float)) for val in item):
            return jsonify({"error": f"All feature values must be numeric (problem at index {i})"}), 400

    input_features = np.array(features)
    
    # Check if input is a single sample (1 row)
    if len(input_features) == 1:
        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)[0]
        confidence = float(probabilities[prediction])
        return jsonify({
            "prediction": int(prediction),
            "confidence": confidence
        })
    else:
        predictions = model.predict(input_features)
        return jsonify({
            "predictions": predictions.tolist()
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) #check your port number ( if it is in use, change the port number)
