# Packages Imports
import base64
import json
from subprocess import call

import numpy as np
from flask import Flask, flash, redirect, request, send_from_directory, url_for
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

import cv2
from places_utils import preprocess_input

# define server app and model
app = Flask(__name__)
model = None



# End-point to upload an image and call predict class for it
# it upload an image encoded to base64. image string is place in request headers with key of 'file'
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    ''' uploads image and call predict for it.'''
    if request.method == 'POST':
        # file = request.json['headers']['file']
        file = request.headers.get('file')
        imgdata = base64.b64decode(file)
        filename = 'imageToPredict.jpg'
        with open('uploads/'+filename, 'wb') as f:
            f.write(imgdata)
        f = predict('uploads/imageToPredict.jpg')
        return f



'''TODO: Use image without saving in disk'''
# def data_uri_to_cv2_img(encoded_data):
#     # encoded_data = uri.split(',')[1]
#     imgdata = base64.b64decode(encoded_data)
#     # nparr = np.fromstring(imgdata, np.uint8)
#     nparr = np.asarray(imgdata, dtype=np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     print(type(img))
#     cv2.imshow('image', img)
#     # return img


labels = ['Architect Campus', 'Buffet',
              'Computer Campus', 'Culture house', 'Field', 'Self']

# End-point to predict last uploded image
def predict(imgaddr):
    ''' predicts the last uploaded image an returns a string at last containing classes probability.'''
    global model
    img = cv2.imread(imgaddr)
    h, w, c = img.shape
    if w > h: # rotate image if it's in wrong orientation
      # rotation is done by ImageMagick so it sohuld be installed
      call(['mogrify', '-rotate', '90', 'uploads/imageToPredict.jpg'])
    img = image.load_img(imgaddr, target_size=(108, 192))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    if not model:
        print('------- loading model')
        model = load_model('PF-50-fixed 24-3-97.h5')
    features = model.predict(x)
    predicts = []
    for i, p in enumerate(features[0]):
        item = '%s Probability: %f' % (labels[i], p)
        predicts.append(item)
    predicts_string = '\n'.join(predicts)
    return predicts_string



# end-point to get the last image sent to predict
@app.route('/imagetopredict')
def uploaded_file():
    # images sent overwrite each other so there is only one image to get
    return send_from_directory('uploads/', 'imageToPredict.jpg')



# End-point to predict again last uploded image
@app.route('/predictagain')
def predict_again():
    f = predict('uploads/imageToPredict.jpg')
    return f


# RUN THE SERVER THING
if __name__ == '__main__':
    app.secret_key = 'abcakjlc-b@weubi_2b3!2@'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=8080)
