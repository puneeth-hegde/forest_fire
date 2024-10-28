from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd

# Load the model and training columns
with open("forest_fire_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("training_columns.pkl", "rb") as f:
    training_columns = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame([data])

    # Add missing columns with a default value of 0
    for col in training_columns:
        if col not in data_df.columns:
            data_df[col] = 0

    # Ensure the columns are in the same order
    data_df = data_df[training_columns]

    # Make the prediction
    prediction = model.predict(data_df)
    return jsonify({'fire_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
