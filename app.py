from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('models/heart_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form values
        values = [
            int(request.form['Sex']),
            int(request.form['ChestPainType']),
            int(request.form['RestingECG']),
            int(request.form['ExerciseAngina']),
            int(request.form['FastingBS']),
            int(request.form['RestingBP']),
            int(request.form['Cholesterol']),
            int(request.form['MaxHR']),
            float(request.form['Oldpeak']),
            int(request.form['ST_Slope']),
            int(request.form['trestbps']),
            int(request.form['thalach']),
            int(request.form['ca']),
            int(request.form['thal']),
            int(request.form['Merged_Age']),
        ]
        
        # Scale & predict
        input_data = scaler.transform([values])
        result = model.predict(input_data)[0]
        prediction = "✅ Disease Detected!" if result == 1 else "🫀 No Disease Detected."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
