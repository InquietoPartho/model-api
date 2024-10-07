from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model (make sure your .keras model file is in the correct path)
model = tf.keras.models.load_model("./models/trained_plant_disease_model.keras")

# Function for prediction
def model_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((128, 128))
    input_arr = np.array(image)  # Convert to numpy array
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Flask route to handle the POST request
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image file as bytes
        image_bytes = file.read()
        
        # Call the prediction function
        prediction_idx = model_prediction(image_bytes)
        
        # Map index to class name
        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                       'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                       'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                       'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                       'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                       'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                       'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                       'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                       'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                       'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                       'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                       'Tomato___healthy']
        
        # Get the class name from prediction and remove underscores
        predicted_class = class_names[prediction_idx].replace('___', ' - ').replace('_', ' ')
        
        return jsonify({"prediction": predicted_class}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
