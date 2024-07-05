from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Example dictionary with class names and image file paths
classes_dict = {
    'Daisy': 'static/daisy[0].jpg',
    'Dandelion': 'static/dandelion[1].jpg',
    'Rose': 'static/rose[2].jpg',
    'Sunflowers': 'static/sunflowers[3].jpg',
    'Tulips': 'static/tulips[4].jpg'
}

# Dictionary to map model prediction to class names
dic = {0: 'Daisy', 1: 'Dandelion', 2: 'Rose', 3: 'Sunflowers', 4: 'Tulips'}

model = load_model('resnet.h5')
model.make_predict_function()

def predict_label(img_path):
    """Function to predict class for a given image path."""
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) 
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    predicted_class_index = np.argmax(p)  # Convert the prediction to an index
    return dic[predicted_class_index]  # Use the index to get the class name

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    predictions = {class_name: predict_label(img_path) for class_name, img_path in classes_dict.items()}
    return render_template("index.html", predictions=predictions, classes_dict=classes_dict)

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        if img:
            img_path = os.path.join('static', img.filename)
            img.save(img_path)
            p = predict_label(img_path)
            return render_template("index.html", prediction=p, img_path=img_path)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
