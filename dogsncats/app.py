# Import library for file and folder manipulation
import os
# Import Flask for website rendering
from flask import Flask, render_template, request, redirect, flash, send_from_directory
# For extra safety secure_filename
from werkzeug.utils import secure_filename
# Import numpy array library
import numpy as np
# Import Tensorflow for Keras, to build Deep Learning models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

# Path to upload images to
UPLOAD_FOLDER = 'static/uploads/'

# Path that contains the model
MODEL_FOLDER = 'static/model/model_epochs.h5'

# Maximum size of image to upload
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024

# Possible extension types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to restrict file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load model and classify the uploaded image, returns the score for 'dogness'
def prediction(filename):
    model = load_model(MODEL_FOLDER)
    image_size = (150, 150) # same size as in my model
    # load image
    img = image.load_img(os.path.join(UPLOAD_FOLDER, filename), target_size=image_size)
    # convert image to array
    img_array = image.img_to_array(img)
    # create batch axis
    img_array = tf.expand_dims(img_array, 0)
    # call model for prediction
    predictions = model.predict(img_array)
    score = predictions[0]
    return score

# About page
@app.route("/about")
def about():
    return render_template('about.html', title='About')

# Classify page on classify request
@app.route('/classify')
def classify():
    return render_template('classify.html',title='Classify')

# Home page
@app.route('/home')
@app.route('/')
def home():
    return render_template('home.html')

# Home page on action
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        result = prediction(filename)
        pred_prob = result.item()  # the model score returns a list, X.item() returns the value in that list

        if pred_prob > 0.5:
            label = 'dog'
        else:
            label = 'cat'
        acc_dog = np.round(pred_prob * 100, 2)
        acc_cat = np.round((1 - pred_prob) * 100, 2)

        return render_template('classify.html', filename=filename, label=label,
                               acc_dog=acc_dog, acc_cat=acc_cat)
    else:
        flash('Allowed image types are: png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')