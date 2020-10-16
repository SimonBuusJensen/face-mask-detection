
from flask import Flask, request, jsonify
from flask_cors import CORS 
from time import time 
import datetime
from model.model import Model
from utilities import NpEncoder, get_uptime, configure_app
from config import PROJECT_NAME, HOST, PORT, THREADED

model = Model()


# --- Welcome to your Emily API! --- #


# See the README for guides on how to test it.
# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default. 
# Make sure to restrict access below to origins you
# trust before deploying your API to production.
app = configure_app(PROJECT_NAME, cors={ 
    r'/api/*': { 
        "origins": "*" 
    }
})

@app.route('/api')
def hello(): 
    return f'The {PROJECT_NAME} API is running (uptime: {get_uptime()}'


@app.route('/api/predict', methods=['POST'])
def predict():
    
    sample = request.form['sample']
    prediction = model.predict(sample)
    print(prediction)
    return jsonify({
        'sample': sample,
        'prediction': float(prediction)
    })
    


@app.route('/api/health')
def healthcheck(): 
    return jsonify({
        'uptime': get_uptime(),
        'status': 'RUNNING',
        'host': HOST, 
        'port': PORT,
        'threaded': THREADED
    })

if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT, threaded=THREADED)

