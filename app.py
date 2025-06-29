from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[field]) for field in [
            'danceability', 'energy', 'loudness', 'acousticness',
            'valence', 'tempo', 'explicit', 'speechiness',
            'instrumentalness', 'liveness','key','mode'
        ]]
       
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
    
        return render_template('index.html', prediction_text=f"Predicted Popularity: {prediction}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)