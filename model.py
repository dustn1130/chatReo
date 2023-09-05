from flask import Flask, request, render_template, jsonify
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
import io  # Add this line to import the 'io' module
from flask import Flask, request, render_template, jsonify, redirect, url_for

app = Flask(__name__)
model = load_model('C:/Users/2002d/Documents/Dr.Reo/model/my_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img = uploaded_file.read()  # Extract image data from FileStorage
            img = image.load_img(io.BytesIO(img), target_size=(150, 150))  # Load image from image data
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            result = "강아지" if classes[0] > 0 else "고양이"
            return jsonify({"result": result})

    return render_template('model.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

if __name__ == '__main__':
    app.run(debug=True)