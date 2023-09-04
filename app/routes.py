from flask import redirect, request, render_template, jsonify
from . import application
import json
from app.ari import scheme_classification


@application.route('/', methods=['GET', 'POST'])
def amf_schemes():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        ff = open(f.filename, 'r')
        content = json.load(ff)
        # Classify the inference relations into 20 different argumentation schemes.
        response = scheme_classification(content)
        return jsonify(response)
    elif request.method == 'GET':
        return render_template('docs.html')
 
 
