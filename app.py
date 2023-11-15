import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Carrega o modelo atualizado para cafeína
model = pickle.load(open("caffeine_model.pkl", "rb"))

# Carrega os dados de cafeína
caffeine_data = pd.read_csv('arquivos-csv\caffeine.csv')
names = dict(zip(caffeine_data['type'], caffeine_data['type']))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Certifique-se de incluir todas as features necessárias
    features = [float(request.form['volume']),
                float(request.form['calories']),
                float(request.form['caffeine'])]
    final_features = [np.array(features)]
    pred = model.predict(final_features)
    output = names[pred[0]]
    return render_template("index.html", prediction_text="Tipo: " + output)

@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    # Certifique-se de incluir todas as features necessárias
    pred = model.predict([np.array([float(data['volume']),
                                    float(data['calories']),
                                    float(data['caffeine'])])])
    output = names[pred[0]]
    return jsonify(output)
