import os,io
import flask
import json
from flask import Flask, request, flash,redirect, url_for, send_from_directory, render_template
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import  load_img, img_to_array
from keras.preprocessing import image
from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
from PIL import Image
from app import app 
import cv2


IMAGE_SIZE = (256,512)  ## Based on the file size
n_classes = 8
image_folder='test_data\images'
mask_folder ='test_data\masks'
iou_score=  sm.metrics.iou_score
meanIOU = tf.keras.metrics.MeanIoU(num_classes=n_classes)
dice_loss = sm.losses.DiceLoss()
global model


# Initializing flask app
app = Flask(__name__, template_folder='templates')  ## To upload files to folder
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

#model = tf.keras.models.load_model('bestModel_Unet_sansAug.h5',
            #custom_objects={'dice_loss': dice_loss ,'iou_score': iou_score, 'mean_IoU': meanIOU, 'dice_coeff': dice_coeff})

                                                             

@app.route("/", methods=['GET'])
def index():
    # afficher le formulaire
    return render_template('index.html', image_name='',   result = False)

@app.route("/result", methods=['POST'])
def get_result():
    
   
    result = request.form
    id_image = result['id_image'] # récupèrer l'id de l'image 
    
    if len( id_image) > 0:
        image_name= id_image+'.png' 
        image_path = str('./static/test_data/images/'+image_name)

        image = img_to_array(load_img(image_path,target_size=(256,256,3)))/255.
        mask_path = str('./static/test_data/masks/'+image_name)
        trueMask = img_to_array(load_img(mask_path, target_size=(256,512), color_mode="grayscale"))
        trueMask = np.squeeze(trueMask)
        plt.imsave('./static/outputs/trueMask.png',  trueMask)
        
        # The mask our model predicts
        img = np.expand_dims(image,axis=0)
        pred_mask=model.predict(img)
        pred_mask = np.argmax(pred_mask, axis=-1)
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        pred_mask = np.squeeze(pred_mask)
        # Use interpolation inter_nearest to use integer with cv2
        pred_mask = cv2.resize(pred_mask, dsize=(512, 256), interpolation=cv2.INTER_NEAREST)
        plt.imsave('./static/outputs/predictedMask.png',  pred_mask)
        
        return render_template("result.html",image_name=image_name, result = True) 
    else:
        flash(u'Identifiant image erroné.')
        return render_template('index.html',image_name='', result = False)
    
    #return render_template('result.html', id_image=id_image)
    
# Running the api
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')