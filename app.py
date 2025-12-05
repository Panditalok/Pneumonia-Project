from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained AI model
# Make sure training finished and this file exists!
model = load_model('pneumonia_model.h5')

# Ensure uploads folder exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        return "PNEUMONIA DETECTED", "danger"
    else:
        return "NORMAL (HEALTHY)", "success"

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    img_path = None
    css_class = ""
    
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            # Save the file to static/uploads
            filename = img_file.filename
            file_path = os.path.join('static/uploads', filename)
            img_file.save(file_path)
            
            # Get prediction
            result, css_class = predict_image(file_path)
            img_path = file_path
            
    return render_template('index.html', result=result, img_path=img_path, css_class=css_class)

if __name__ == '__main__':
 app.run(debug=True, host='0.0.0.0', port=5001)