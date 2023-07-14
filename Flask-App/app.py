from flask import Flask, request, render_template, flash
from PIL import Image
import torch
import os
from werkzeug.utils import secure_filename
import pandas as pd
import sys
sys.path.append('../')
import MainFunctions as netfuncs
from MainFunctions import importimg, buildmodelFromParams
from AlgorithmDefinition import MatchEGMAlgorithm

# declare constants
HOST = '0.0.0.0'
PORT = 5000

# initialize flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'
app.config['UPLOAD_FOLDER'] = "static/images"

currentint = int(open('static/currentint.txt', 'r').readline())
df = pd.read_csv("static/logging.csv") if (os.path.isfile("static/logging.csv")) else pd.DataFrame()
NETS = "static/neuralnets/netparams.csv"
    
def loadmodels(netparams_loc):
    """Function for instantiating and loading prediction models

    Args:
        netparams_loc (str): location where model weights are stored

    Returns:
        dict: dictionary containing loaded prediction models
    """
    models = {}
    netparams = pd.read_csv(netparams_loc)
    for idx, netparam in netparams.iterrows():
        netparam = netparam.to_dict()
        model = buildmodelFromParams(netparams=netparam, load=True)
        models[model.modelname] = model     
    models["MatchEGMAlgorithm"] = MatchEGMAlgorithm()
    return models

#loading neural nets
models = loadmodels(NETS)

@app.route('/', methods=['GET', 'POST'])
def root():
    """Main route for main page

    Returns:
        str: rendered main page
    """
    global currentint, df    

    if request.method == 'POST':
        image = secure_filename(request.files["image"].filename)
        template = secure_filename(request.files["template"].filename)
        if image == '' or template == '':
            flash("There is data missing!", "error")
            return render_template("index.html")

        image = f"{currentint}_img_{image}"
        template = f"{currentint}_temp_{template}"
  
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], image)
        template_path = os.path.join(app.config['UPLOAD_FOLDER'], template) 
        label = request.form["text"]
        black = request.form["black"] =='True'

        request.files["image"].save(img_path)
        request.files["template"].save(template_path)

        netpreds = evalimgs(template_path, img_path, black, models)
        results = {"label" : label, "black" : black} | netpreds

        messages = {"success_text" : "File Sucessfully Send", 
                    "results" :  results
                    }

        results = {"img" : img_path, "template" : template_path} | results
        
        df = df.append(results, ignore_index=True)
        df.to_csv("static/logging.csv", index=False)
        
        currentint += 1
        with open('static/currentint.txt', 'w') as f: f.write('%d' % currentint)

        flash(messages, "success")
    return render_template("index.html")

@app.route('/results', methods=['GET'])
def results():
    """Route to results page

    Returns:
        str: rendered results page
    """
    df = pd.read_csv("static/logging.csv").sort_index(ascending=False)
    return render_template("results.html",column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip)


def evalimgs(template_path, img_path, black, models):
    """Function for executing image evaluation

    Args:
        template_path (str): local path to uploaded template
        img_path (str): local path to uploaded match image
        black (bool): boolean for background of signal, mostly black or white
        models (dict): dict containing prediction models

    Returns:
        dict: dict containing predictions from each model
    """
    invert = False if black else True
    template = importimg(template_path, invert)
    img = importimg(img_path, invert)
    preds = {}
    
    for model in models.values():
        pred = model.predict(img, template)
        preds[model.modelname] = pred
        
    return preds


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    # run web server
    app.run(host=HOST,
            debug=True,
            port=PORT)
