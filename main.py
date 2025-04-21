from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        gmail = request.form['gmail']

        # Collect features
        inputs = [
            int(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            int(request.form['trestbps']),
            int(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            int(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]

        final_input = np.array([inputs])  # Ensure it's in 2D array shape

        # Predict class and probability
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]
        confidence = round(probability * 100, 2)

        if prediction == 1:
            result = f"⚠️ Heart Disease Detected with a {confidence}% confidence."
            recommendation = (
                "Please consult a cardiologist as soon as possible. "
                "Consider reducing cholesterol, quitting smoking, exercising regularly, and maintaining a heart-healthy diet."
            )
        else:
            result = f"✅ No Heart Disease Detected. Confidence: {100 - confidence}%"
            recommendation = (
                "Great job! Keep up a healthy lifestyle with regular exercise, a balanced diet, and stress management."
            )

        return render_template("result.html", name=name, result=result, recommendation=recommendation)

    except Exception as e:
        return f"Error: {str(e)}"

# Start the app (use a port that works with Replit)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

