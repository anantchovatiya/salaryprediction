from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS globally

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


model = pickle.load(open("salary_model.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    skills = data.get("skills", [])
    experience = data.get("experience", 0)

    input_skills = mlb.transform([skills])
    experience_scaled = scaler.transform([[experience]])
    input_data = np.hstack((input_skills, experience_scaled))

    predicted_salary = model.predict(input_data)[0]
    return jsonify({"predicted_salary": round(predicted_salary, 2)})

if __name__ == "__main__":
    app.run(debug=True)
