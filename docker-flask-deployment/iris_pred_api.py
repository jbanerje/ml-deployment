#------------------------------------------------------------------------------------------------------------------------------#
# PROJECT    : SAMPLE - IRIS PREDICTION API                                             
# OBJECTIVE  : 
# DATE       : 11/3/2021
# WRITTEN BY : JAGANNATH BANERJEE
#-------------------------------------------------------------------------------------------------------------------------------#

# Import Libraries
from schema import Schema, And, SchemaError
from flask import Flask, request, jsonify, Response
import pandas as pd
import datetime
import json
import pickle

app = Flask(__name__)

@app.route('/iris-pred-api', methods=['POST'])

def predict_iris_species():

    data_dict = {}

    timestamp_for_file = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    model = pickle.load(open('./model/iris_model.pkl', 'rb'))

    predict_map = {0:'setosa', 1:'virginica', 2:'versicolor'}

    schema = Schema(
                        {
                            'sepal_length'  : And(float),
                            'sepal_width'   : And(float),
                            'petal_length'  : And(float),  
                            'petal_width'   : And(float)
                        }
                )
    
    # get the json string
    req_data = request.get_json()
        
    try:
        schema.validate(req_data)

        pred_rec = pd.DataFrame([req_data])
        prediction  = predict_map[model.predict(pred_rec)[0]]
        probability = (model.predict_proba(pred_rec).max() * 100).astype(int).astype(str)

        response_dict = {'Prediction': prediction , 'Confidence' : probability}
        
        resp = json.dumps(response_dict, ensure_ascii = False)
        
        response_json = Response(response     = resp, 
                                 status       = 200,
                                 content_type = "application/json; charset=utf-8"
                                )

    except SchemaError as json_sch_err:

        response_dict = {'Prediction': "Schema Error"}

        resp = json.dumps(response_dict, ensure_ascii = False)

        response_json = Response(response     = resp, 
                                 status       = 400,
                                 content_type = "application/json; charset=utf-8"
                                )
        print(json_sch_err)
    return response_json

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)