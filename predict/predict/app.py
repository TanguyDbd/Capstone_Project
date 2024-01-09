from flask import Flask
from predict.predict import run as predict_run
import json

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    artefacts_path = 'C:/Users/tangu/OneDrive/Documents/5A/From_Poc_to_Prod/poc-to-prod-capstone/train/data/artefacts/2024-01-09-12-51-13'
    model = predict_run.TextPredictionModel.from_artefacts(artefacts_path)

    # Text for prediction
    user_text = "This course was very interesting, tennis is the best sport ever"

    # Perform prediction
    predictions = model.predict([user_text])

    results = [model.labels_to_index[str(idx)] for idx in predictions[0]]
    result_json = json.dumps({'input_text': user_text, 'predictions': results})
    return result_json


if __name__ == '__main__':
    app.run(debug=True)
    