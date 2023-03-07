import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle as pkl

app_api = Flask(__name__)

model = pkl.load(open('model.pkl', 'rb'))

@app_api.route('/predict', methods = ["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({"Prediction": list(prediction)})

if __name__ == "__main__":
    app_api.run(debug=True, port=8888)