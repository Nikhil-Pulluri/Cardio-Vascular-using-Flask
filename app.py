import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier  # Import XGBoost

app = Flask(__name__)

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=column_names)

# Replace missing values ('?') with NaN
data.replace('?', pd.NA, inplace=True)

# Convert columns with missing values to numeric
data['ca'] = pd.to_numeric(data['ca'])
data['thal'] = pd.to_numeric(data['thal'])

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
data[['ca', 'thal']] = imputer.fit_transform(data[['ca', 'thal']])

# Encode categorical variables if needed
data['sex'] = data['sex'].astype(int)
data['cp'] = data['cp'].astype(int)
data['fbs'] = data['fbs'].astype(int)
data['restecg'] = data['restecg'].astype(int)
data['exang'] = data['exang'].astype(int)
data['slope'] = data['slope'].astype(int)

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Binarize the target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': float(request.form['trestbps']),
            'chol': float(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': float(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': float(request.form['ca']),
            'thal': float(request.form['thal'])
        }

        user_input = pd.DataFrame([user_data])
        user_input = scaler.transform(user_input)

        risk_score = model.predict(user_input)
        risk_probability = model.predict_proba(user_input)

        if risk_score[0] == 1:
            result = f"Based on the provided data, there is a risk of cardiovascular disease with a probability of {risk_probability[0][1]*100:.2f}%."
        else:
            result = f"Based on the provided data, there is a low risk of cardiovascular disease with a probability of {risk_probability[0][0]*100:.2f}%."

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
