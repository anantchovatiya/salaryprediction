from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS


model = pickle.load(open('salary_model.pkl', 'rb'))
mlb = pickle.load(open('mlb.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
@app.route('/')
def home():
    return """
    <h1>Welcome to the Salary Prediction API</h1>
    <p>Use the /predict endpoint to get salary predictions based on skills and experience.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    skills = data.get('skills', [])
    experience = data.get('experience', 0)

    # Transform input skills
    input_skills = mlb.transform([skills])
    experience_scaled = scaler.transform([[experience]])
    input_data = np.hstack((input_skills, experience_scaled))

    # Predict salary
    predicted_salary = model.predict(input_data)[0]

    return jsonify({'predicted_salary': round(predicted_salary, 2)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Get Render-assigned PORT
    app.run(host='0.0.0.0', port=port)
