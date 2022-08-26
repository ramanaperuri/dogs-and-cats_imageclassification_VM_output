# Import library for file and folder manipulation
import os
# Import Flask for website rendering
from flask import Flask, render_template, request, send_from_directory
# Import numpy array library
import numpy as np
# Import Tensorflow for Keras, to build Deep Learning models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras_preprocessing import image

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

# static folder
STATIC_FOLDER = 'static'
# Path to the uploads folder, to store the images for prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder that contains the model CNN
MODEL_FOLDER = STATIC_FOLDER + '/model'

# Function to load my own model and classify an uploaded image
def prediction_my_model(filename):
    model = load_model(MODEL_FOLDER + "/model_epochs.h5")
    image_size = (150, 150) # same as in my model
    img = keras.preprocessing.image.load_img(filename, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    return score

# Function to load model based on MobileNetV2 and classify an uploaded image
def prediction_MobNetv2(filename):
    model = load_model(MODEL_FOLDER + "/MobNetv2_10epochs_fine_tuning.h5")
    image_size = (150, 150) # same as in my model
    img = keras.preprocessing.image.load_img(filename, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predictions = tf.nn.sigmoid(predictions)  # use a sigmoid to get scores, model last layer is logit!!
    score = predictions[0]
    return score

# About page
@app.route("/about")
def about():
    return render_template('about.html', title='About')

# Home page
@app.route("/home")
@app.route('/')
def home():
    return render_template('home.html')

# Upload file and classify as Cat or Dog
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)
        # Get predictions from the respective models
        result_mine = prediction_my_model(fullname)
        result_MobNet = prediction_MobNetv2(fullname)
        # Retrieve from list
        pred_prob_mine = result_mine.item() # the model score returns a list, X.item() returns the value in that list
        pred_prob_MobNet = result_MobNet#.item()

        # Get label and scores for my model
        if pred_prob_mine > 0.5:
            label_mine = 'dog'
        else:
            label_mine = 'cat'
        acc_dog_mine = np.round(pred_prob_mine * 100, 2)
        acc_cat_mine = np.round((1 - pred_prob_mine) * 100, 2)

        # Get label and scores for model based on MobileNetV2
        if pred_prob_MobNet > 0.5:
            label_MobNet = 'dog'
        else:
            label_MobNet = 'cat'
        acc_dog_MobNet = np.round(pred_prob_MobNet * 100, 2)
        acc_cat_MobNet = np.round((1 - pred_prob_MobNet) * 100, 2)

        # Return labels and scores for 'classify.html'
        return render_template('classify.html', image_file_name=file.filename, label1=label_mine, label2=label_MobNet,
                               acc_dog1 =acc_dog_mine, acc_cat1=acc_cat_mine,
                                                                acc_dog2 =acc_dog_MobNet, acc_cat2=acc_cat_MobNet)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)