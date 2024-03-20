# flask
from flask import Flask,request,jsonify
from werkzeug.utils import secure_filename

# converting image
import base64
import re
from PIL import Image
import io
import os

# keras
import tensorflow as tf
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import numpy as np


app = Flask(__name__)

# load trained model
model = tf.lite.Interpreter(model_path=".venv/sdgp/model.tflite")
model.allocate_tensors()


# define prediction route
@app.route("/predict",methods=["POST"])
def predict_route():
    if "image" not in request.json:
        print("No image data found in request")
        return jsonify("No image data found")


# Preprocess your image
def model_predict(img_path,model):
    from keras.preprocessing import image
    img_path = '.venv/sdgp/cc863e23-d7a4-46f6-b4d4-41d5c73caeac.jpeg'
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_array = img_array / 255.0

    # converting image data
    # Extract base64 image data
    base64_img = request.json["image"]
    base64_data = re.sub("^data:image/.+;base64,', '', base64_img")
    # Decode base64 image data
    try:
        image_data = base64.b64decode(base64_data)
    except Exception as e:
        print(f"error decoding base64 data:, {e}")
        return jsonify({"error": "error decoding base64 data"})

    # convert binary data to image
    binary_img = ["image_data"]
    image = Image.open(io.BytesIO(binary_img))
    image.show()

    # save the image temporarily
    temp_img_path = ".venv/sdgp/cc863e23-d7a4-46f6-b4d4-41d5c73caeac.jpeg"
    # Save the decoded image data to a temporary file
    with open(temp_img_path, 'wb') as f:
        f.write(image_data)
    # Process the image as needed
    image = Image.open(temp_img_path)
    # Delete the temporary file
    os.remove(temp_img_path)


    # Make predictions
    predictions = model.predict(img_array)

    # process the result to user
    category_labels=[['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
                      'Atopic Dermatitis Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos',
                      'Exanthems and Drug Eruptions', 'Herpes HPV and other STDs Photos',
                      'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
                      'Melanoma Skin Cancer Nevi and Moles', 'Poison Ivy Photos and other Contact Dermatitis',
                      'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors',
                      'Systemic Disease','Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives',
                      'Vascular Tumors', 'Vasculitis Photos','Warts Molluscum and other Viral Infections']]

    # decode the predictions
    predicted_category_index = np.argmax(predictions[0])
    predicted_category_label = category_labels[predicted_category_index]

    # Return the processed result as a response
    return jsonify({"result": predicted_category_label})


if __name__ == '__main__':
    app.run(port=3000, debug=True)
