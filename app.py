from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import torch
from pathlib import Path
from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy
import warnings
import os
import PIL
import fastai
from fastai.vision import *
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pydicom
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)



# path = Path("path")
classes = ['ADENOSIS', 'Ductal Carcinoma', 'FIBRO ADENOMA', 'MUCINOUS CARCINOMA', 'TUBULAR ADENOMA']
# data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
# learn = create_cnn(data2, models.resnet34)
# learn.load('stage-2')




def model_predict(img_path, ensemble):
    """
       model_predict will return the preprocessed image
    """
    defaults.device = torch.device('cpu')
    # path = Path('path/models')
    # print(path)
    alexnet = load_learner('path/models/alexnet_model.pkl')
    vgg = load_learner('path/models/vgg_model.pkl')
    resnet = load_learner('path/models/resnet_model.pkl')

    if(ensemble == "0"):
        return ensemble_classification(img_path,resnet,vgg,alexnet)

    else:
        return stacking_ensemble_classification(img_path,resnet,vgg,alexnet)
         
    

def stacking_ensemble_classification(image,resnet_learner, vgg_learner, alexnet_learner):
    ens=[resnet_learner ,vgg_learner, alexnet_learner ]
    imga = PIL.Image.open(image)
    img_tensor= TensorImage(image2tensor(imga))
    img= PILImage.create(img_tensor)
    ens_test_preds = [] ## Creating a list of predictions 
    for mdl in ens:
        preds = mdl.predict(img)
        # print(np.array(preds))
        ens_test_preds.append(preds)
    
    ens_preds =sum(value[2] for value in ens_test_preds) / len(ens_test_preds)
    predicted_class = torch.argmax(ens_preds).item()
    return classes[predicted_class]



def ensemble_classification(test_img, resnet_learner, vgg_learner, alexnet_learner):
    imga = PIL.Image.open(test_img)
    img_tensor= TensorImage(image2tensor(imga))
    img= PILImage.create(img_tensor)
    resnet_prediction = resnet_learner.predict(img)
    vgg_prediction = vgg_learner.predict(img)
    alexnet_prediction = alexnet_learner.predict(img)
    print('--------------resnet----------------------')
    print(resnet_prediction[1])
    print('----------vgg--------------------------')
    print(vgg_prediction[1])
    print('------------------------------------')
    print(alexnet_prediction[1])
    print('---------------------------------------------')

    list = []
    list.append(resnet_prediction)
    list.append(vgg_prediction)
    list.append(alexnet_prediction)
  
    

    # sum_pred = resnet50_prediction[2] + resnet34_prediction[2] + vgg_prediction[2] + vgg_prediction[2]
    prediction_most = most_frequent(list)

    

    print('-----------Ensemble-------------------------')
    print(prediction_most)
    # print()
    #prediction results
    # predicted_label = torch.argmax(prediction_most[2]).item()
    
    return prediction_most[0]


def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        ensemble_type = request.form['ensemble']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # # Make prediction
        preds = model_predict(file_path, ensemble_type)
        return preds
    return None


if __name__ == '__main__':
    
    app.run(host='localhost', port='5002',debug=False)


