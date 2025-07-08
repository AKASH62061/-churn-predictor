from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define valid features
VALID_FEATURES = [
    'Contract_Month-to-month',
    'TotalCharges',
    'InternetService_Fiber optic',
    'MonthlyCharges',
    'PaymentMethod_Electronic check',
    'MultipleLines_No',
    'OnlineBackup_No'
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        # Convert form input into the expected DataFrame
        input_data = {
            'Contract_Month-to-month': 1 if form['Contract'] == 'Month-to-month' else 0,
            'TotalCharges': float(form['TotalCharges']),
            'InternetService_Fiber optic': 1 if form['InternetService'] == 'Fiber optic' else 0,
            'MonthlyCharges': float(form['MonthlyCharges']),
            'PaymentMethod_Electronic check': 1 if form['PaymentMethod'] == 'Electronic check' else 0,
            'MultipleLines_No': 1 if form['MultipleLines'] == 'No' else 0,
            'OnlineBackup_No': 1 if form['OnlineBackup'] == 'No' else 0
        }

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        result = "Churn" if prediction == 1 else "No Churn"
        return render_template('index.html', prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
